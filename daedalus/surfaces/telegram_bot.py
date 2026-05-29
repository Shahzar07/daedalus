"""The Telegram surface — reach Daedalus from your phone.

Like every surface, this is a thin shell around the *same* :class:`~daedalus.core.loop.Agent`
the terminal uses: memory, skills, budget, guardrails, and the kill switch all behave
identically. What this module adds is the Telegram-specific plumbing:

  * **Per-chat conversation history.** One bot can serve several chats; each keeps its
    own thread. We swap ``agent.history`` in and out per chat under a single lock, so two
    chats (or a chat and a scheduled job) never interleave into the one agent.
  * **Inline-button approvals.** When a tool needs approval (``ask`` policy), the bot
    sends an Allow/Deny keyboard and *awaits the tap* — the same futures-based pattern the
    web surface uses, so destructive actions are gated on your phone too.
  * **An allowlist.** A public bot username can be messaged by anyone, so by default we
    only answer chat IDs in ``TELEGRAM_ALLOWED_CHAT_IDS``. With none set we run in "open
    mode" and say so loudly at startup.
  * **Scheduler delivery.** If the ``[scheduler]`` extra is present, user-defined jobs run
    here and their results are pushed to your chat — a recurring task that texts you.

python-telegram-bot lives behind the optional ``[telegram]`` extra; this module is
imported lazily by ``dae telegram`` so the core install stays slim.
"""

from __future__ import annotations

import asyncio
import contextlib

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..config import Settings
from ..core.loop import Agent
from ..safety import ApprovalRequest

# Telegram rejects messages longer than 4096 characters; leave headroom for markup.
_MAX_MESSAGE = 3900


def chunk_message(text: str, limit: int = _MAX_MESSAGE) -> list[str]:
    """Split a long reply into Telegram-sized pieces, preferring line boundaries."""
    text = text or "(no output)"
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # a single monster line: hard-split it
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


_HELP = (
    "*Daedalus*\n"
    "Send me a message and I'll work on it. Commands:\n"
    "/help — this help\n"
    "/memory — what I remember\n"
    "/skills — playbooks I can use\n"
    "/jobs — your scheduled tasks\n"
    "/budget — today's spend + kill-switch state\n"
    "/stop — engage the kill switch (halts every run)\n"
    "/resume — release the kill switch\n"
    "/reset — clear this chat's history"
)


class TelegramBot:
    """Owns the bot application and bridges Telegram updates to the agent."""

    def __init__(self, agent: Agent, settings: Settings):
        self.agent = agent
        self.settings = settings
        self.allowlist = settings.telegram_allowlist()
        # One lock serializes all agent use (chats + scheduled jobs); per-chat history
        # lives here and is swapped onto the agent for the duration of each turn.
        self.run_lock = asyncio.Lock()
        self.histories: dict[int, list] = {}
        # Approval futures keyed by id, resolved when the user taps Allow/Deny.
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._active_chat_id: int | None = None
        self._last_chat_id: int | None = None
        self._bot = None  # the live telegram Bot, set in run_telegram/on_message
        self.scheduler = None  # set in _post_init if the [scheduler] extra is present

    def authorized(self, chat_id: int) -> bool:
        """True if this chat may use the bot (open mode answers everyone)."""
        return not self.allowlist or chat_id in self.allowlist

    # ---- lifecycle -----------------------------------------------------------

    def build(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self.settings.telegram_bot_token)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )
        app.add_handler(CommandHandler(["start", "help"], self.cmd_help))
        app.add_handler(CommandHandler("memory", self.cmd_memory))
        app.add_handler(CommandHandler("skills", self.cmd_skills))
        app.add_handler(CommandHandler("jobs", self.cmd_jobs))
        app.add_handler(CommandHandler("budget", self.cmd_budget))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CommandHandler("resume", self.cmd_resume))
        app.add_handler(CommandHandler("reset", self.cmd_reset))
        app.add_handler(CallbackQueryHandler(self.on_callback, pattern=r"^approve:"))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message))
        return app

    async def _post_init(self, app: Application) -> None:
        """Start the scheduler once the bot's event loop is running (if available)."""
        try:
            from ..scheduler.heartbeat import Scheduler
            from ..scheduler.jobs import JobStore
        except ImportError:
            return  # [scheduler] extra not installed — bot still works, just no jobs
        store = JobStore(self.settings.dae_home / "jobs.db")

        async def deliver(job, result: str) -> None:
            target = self._delivery_target()
            if target is None:
                return
            header = f"⏰ *scheduled job* `{job.id}` — {job.human}\n_{job.prompt}_\n\n"
            for piece in chunk_message(header + result):
                with contextlib.suppress(Exception):
                    await app.bot.send_message(target, piece, parse_mode="Markdown")

        self.scheduler = Scheduler(self.agent, store, deliver=deliver, run_lock=self.run_lock)
        await self.scheduler.start()

    async def _post_shutdown(self, app: Application) -> None:
        if self.scheduler is not None:
            await self.scheduler.shutdown()

    def _delivery_target(self) -> int | None:
        """Where scheduled-job output goes: a configured chat, else the last to message."""
        if self.allowlist:
            return sorted(self.allowlist)[0]
        return self._last_chat_id

    # ---- approval (inline keyboard) -----------------------------------------

    async def _approve(self, req: ApprovalRequest) -> bool:
        """Guardrails approver: send an Allow/Deny keyboard and await the tap."""
        chat_id = self._active_chat_id
        if chat_id is None:
            return False  # no chat context => fail closed
        approval_id = req.tool + ":" + str(id(req))
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[approval_id] = fut
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Allow", callback_data=f"approve:{approval_id}:yes"),
                    InlineKeyboardButton("⛔ Deny", callback_data=f"approve:{approval_id}:no"),
                ]
            ]
        )
        text = f"⚠ *Approval needed* for `{req.tool}`\n_{req.reason}_\n`{str(req.args)[:300]}`"
        try:
            await self._bot.send_message(
                chat_id, text, reply_markup=keyboard, parse_mode="Markdown"
            )
            return await asyncio.wait_for(fut, timeout=300)
        except Exception:  # noqa: BLE001 - timeout or send failure => deny
            return False
        finally:
            self._pending.pop(approval_id, None)

    async def on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()
        _, approval_id, verdict = query.data.split(":", 2)
        fut = self._pending.get(approval_id)
        if fut is not None and not fut.done():
            fut.set_result(verdict == "yes")
        with contextlib.suppress(Exception):
            await query.edit_message_text(
                f"{'✅ Allowed' if verdict == 'yes' else '⛔ Denied'} `{approval_id.split(':')[0]}`",
                parse_mode="Markdown",
            )

    # ---- message handling ----------------------------------------------------

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        text = (update.message.text or "").strip() if update.message else ""
        if not self.authorized(chat_id):
            await context.bot.send_message(
                chat_id,
                f"Sorry, this Daedalus instance is private. Your chat ID is `{chat_id}` — "
                "add it to TELEGRAM_ALLOWED_CHAT_IDS to get access.",
                parse_mode="Markdown",
            )
            return
        if not text:
            return
        self._last_chat_id = chat_id

        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        # One agent, one lock: load this chat's history, point approvals at this chat,
        # run, then save the history back. Serialized so chats never interleave.
        async with self.run_lock:
            self._bot = context.bot
            self._active_chat_id = chat_id
            self.agent.history = self.histories.get(chat_id, [])
            guardrails = self.agent.guardrails
            previous = guardrails.approver if guardrails is not None else None
            if guardrails is not None:
                guardrails.approver = self._approve
            try:
                answer = await self.agent.run(text)
            except Exception as exc:  # noqa: BLE001 - report, never crash the bot
                answer = f"error: {exc}"
            finally:
                if guardrails is not None:
                    guardrails.approver = previous
                self.histories[chat_id] = self.agent.history
                self._active_chat_id = None

        for piece in chunk_message(answer):
            await context.bot.send_message(chat_id, piece)

    # ---- commands ------------------------------------------------------------

    async def _guard(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
        """Shared authorization check for command handlers; returns the chat id or None."""
        chat_id = update.effective_chat.id
        if not self.authorized(chat_id):
            await context.bot.send_message(chat_id, f"Private instance. Your chat ID: {chat_id}")
            return None
        return chat_id

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is not None:
            await context.bot.send_message(chat_id, _HELP, parse_mode="Markdown")

    async def cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        facts = self.agent.memory.recent(limit=10) if self.agent.memory else []
        body = "\n".join(f"• {f}" for f in facts) if facts else "Nothing remembered yet."
        await context.bot.send_message(
            chat_id, f"*Remembered facts*\n{body}", parse_mode="Markdown"
        )

    async def cmd_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        skills = self.agent.skills.all() if self.agent.skills else []
        body = (
            "\n".join(f"• *{s.name}* — {s.description}" for s in skills[:30])
            if skills
            else "No skills loaded."
        )
        await context.bot.send_message(chat_id, f"*Skills*\n{body}", parse_mode="Markdown")

    async def cmd_jobs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        await context.bot.send_message(chat_id, _render_jobs(self.scheduler), parse_mode="Markdown")

    async def cmd_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        await context.bot.send_message(
            chat_id, _render_budget(self.agent.budget), parse_mode="Markdown"
        )

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        if self.agent.budget is not None:
            self.agent.budget.engage_kill_switch()
        await context.bot.send_message(
            chat_id, "🛑 Kill switch *engaged*. Runs halt until /resume.", parse_mode="Markdown"
        )

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        existed = self.agent.budget.release_kill_switch() if self.agent.budget else False
        msg = "✅ Kill switch released." if existed else "Kill switch was not engaged."
        await context.bot.send_message(chat_id, msg)

    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = await self._guard(update, context)
        if chat_id is None:
            return
        self.histories.pop(chat_id, None)
        await context.bot.send_message(chat_id, "History cleared for this chat.")


def _render_budget(budget) -> str:
    if budget is None:
        return "Budget tracking is off."
    snap = budget.snapshot()
    cap = float(snap["daily_cap_usd"])
    today = (
        f"${snap['daily_cost_usd']:.4f} / ${cap:.2f} cap"
        if cap > 0
        else f"${snap['daily_cost_usd']:.4f} (no daily cap)"
    )
    return (
        "*Budget*\n"
        f"• today: {today}\n"
        f"• session: {snap['session_tokens']} tokens / ${snap['session_cost_usd']:.4f}\n"
        f"• kill switch: {'ENGAGED' if snap['killed'] else 'off'}"
    )


def _render_jobs(scheduler) -> str:
    if scheduler is None:
        return "Scheduler not running (install the `[scheduler]` extra and restart)."
    jobs = scheduler.list_jobs()
    if not jobs:
        return 'No scheduled jobs. Create one with `dae jobs add "<when>" "<prompt>"`.'
    lines = ["*Scheduled jobs*"]
    for j in jobs:
        state = "▶" if j.enabled else "⏸"
        lines.append(f"{state} `{j.id}` — {j.human}\n    _{j.prompt[:80]}_")
    return "\n".join(lines)


def run_telegram(agent: Agent, settings: Settings) -> None:
    """Build the bot and block on long-polling. Raises ``SystemExit`` if no token is set."""
    if not settings.telegram_bot_token:
        raise SystemExit(
            "No Telegram token. Get one from @BotFather, then set TELEGRAM_BOT_TOKEN in .env."
        )
    bot = TelegramBot(agent, settings)
    if not bot.allowlist:
        print(
            "[telegram] WARNING: TELEGRAM_ALLOWED_CHAT_IDS is empty — running in OPEN mode "
            "(anyone who finds the bot can use it). Set it in .env to lock the bot down."
        )
    app = bot.build()
    bot._bot = app.bot  # the approver reaches the bot through here
    print("[telegram] bot is live — message it from Telegram. Ctrl-C to stop.")
    app.run_polling()
