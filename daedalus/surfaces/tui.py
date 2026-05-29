"""The terminal UI — Daedalus's default face.

A Textual app with three live regions and a command line:

  * **Transcript** (left) — the clean conversation: what you said, what Daedalus
    answered.
  * **Activity** (right) — the agent thinking out loud in real time: its reasoning,
    every tool it calls, and every result that comes back.
  * **Stats** (bottom-right) — provider/model and a running token tally for the session.

Nothing here is wired into the loop directly. The app *subscribes to the event bus*
(:mod:`daedalus.core.events`) and redraws as events arrive — the same decoupling the
web trace viewer (M7) and the audit log (M6) will reuse. ``agent.run`` executes inside
a Textual worker so the UI keeps breathing while the agent works.

Slash commands (`/help`, `/memory`, `/skills`, `/budget`, `/stop`, `/resume`, `/reset`,
`/clear`, `/exit`) are handled before anything reaches the model. When a tool that can
change the world needs approval (the `ask` policy), the app raises a modal so you can
allow or deny it without leaving the keyboard.
"""

from __future__ import annotations

import json

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from ..config import Settings
from ..core.events import Event, EventType, get_event_bus
from ..core.loop import Agent
from ..safety import ApprovalRequest

_HELP = """\
[b]Commands[/b]
  [cyan]/help[/]    show this help
  [cyan]/memory[/]  list what Daedalus remembers
  [cyan]/skills[/]  list the skills (playbooks) Daedalus can use
  [cyan]/jobs[/]    list scheduled tasks (manage with `dae jobs ...`)
  [cyan]/budget[/]  show today's spend, session tokens, kill-switch state
  [cyan]/stop[/]    engage the kill switch (halts every run until /resume)
  [cyan]/resume[/]  release the kill switch
  [cyan]/reset[/]   clear conversation history
  [cyan]/clear[/]   clear the transcript pane
  [cyan]/exit[/]    quit (also Ctrl+C)
Anything else is sent to the agent.\
"""


def _short(value: object, limit: int = 120) -> str:
    """Collapse whitespace and clip long values so the Activity pane stays scannable."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ApprovalScreen(ModalScreen[bool]):
    """A blocking yes/no modal for approving a world-changing tool call.

    Returned to the guardrails approver via ``push_screen_wait``: ``True`` allows the
    call, ``False`` denies it. Escape and the Deny button both fail closed.
    """

    BINDINGS = [("y", "allow", "Allow"), ("n", "deny", "Deny"), ("escape", "deny", "Deny")]

    CSS = """
    ApprovalScreen { align: center middle; }
    #dialog { width: 64; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    #buttons { height: auto; align: center middle; padding-top: 1; }
    #buttons Button { margin: 0 2; }
    """

    def __init__(self, tool: str, args: str, reason: str):
        super().__init__()
        self._tool = tool
        self._args = args
        self._reason = reason

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                f"[b yellow]Approval needed[/]\n[b]{self._tool}[/] [dim]({self._reason})[/]"
            )
            yield Static(f"[dim]{self._args}[/]")
            with Horizontal(id="buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class DaedalusApp(App):
    """The Textual application. Holds an :class:`Agent` and renders its event stream."""

    TITLE = "Daedalus"
    SUB_TITLE = "self-hosted AI agent"

    CSS = """
    #body { height: 1fr; }
    #transcript { width: 2fr; border: round $primary; padding: 0 1; }
    #sidebar { width: 1fr; }
    #activity { height: 1fr; border: round $accent; padding: 0 1; }
    #stats { height: auto; border: round $panel-darken-1; padding: 0 1; color: $text-muted; }
    #prompt { dock: bottom; border: tall $primary; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, agent: Agent, settings: Settings):
        super().__init__()
        self.agent = agent
        self.settings = settings
        self._tokens = 0
        self._unsubscribe = lambda: None
        self._worker = None  # the in-flight agent run, if any (for /stop)

    # ---- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield RichLog(id="transcript", markup=True, wrap=True, highlight=False)
            with Vertical(id="sidebar"):
                yield RichLog(id="activity", markup=True, wrap=True, highlight=False)
                yield Static(self._stats_text(), id="stats")
        yield Input(placeholder="Ask Daedalus…  (/help for commands)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#activity", RichLog).border_title = "activity"
        self.query_one("#transcript", RichLog).border_title = "conversation"
        transcript = self.query_one("#transcript", RichLog)
        transcript.write(
            f"[b cyan]Daedalus[/] is ready — provider [b]{self.settings.model_provider}[/], "
            f"model [b]{self.settings.model_name}[/]."
        )
        transcript.write("[dim]Type a message, or /help for commands.[/]")
        # Listen to the agent's nervous system; redraw as events arrive.
        self._unsubscribe = get_event_bus().subscribe(self._on_event)
        # Wire the modal approver into the guardrails so an `ask`-policy tool raises a
        # dialog instead of failing closed (the headless default).
        if self.agent.guardrails is not None:
            self.agent.guardrails.approver = self._approve
        self.query_one("#prompt", Input).focus()

    def on_unmount(self) -> None:
        self._unsubscribe()

    async def _approve(self, req: ApprovalRequest) -> bool:
        """Guardrails approver: raise the modal and wait for the user's decision.

        Runs inside the agent worker (``agent.run`` awaits this), so ``push_screen_wait``
        can block for the result without freezing the UI."""
        return await self.push_screen_wait(
            ApprovalScreen(req.tool, _short(dict(req.args), 200), req.reason)
        )

    # ---- event-bus subscriber (live "streaming" of the agent's steps) --------

    def _on_event(self, ev: Event) -> None:
        """Render one bus event. Runs on the app's loop (the worker shares it), so
        touching widgets here is safe. We skip FINAL — the answer is printed from the
        worker's return value to keep the transcript clean and duplicate-free."""
        activity = self.query_one("#activity", RichLog)
        if ev.type == EventType.THOUGHT:
            activity.write(f"[dim italic]· {_short(ev.data.get('text'))}[/]")
        elif ev.type == EventType.TOOL_CALL:
            activity.write(
                f"[yellow]→ {ev.data.get('name')}[/]([dim]{_short(ev.data.get('args'))}[/])"
            )
        elif ev.type == EventType.TOOL_RESULT:
            activity.write(f"[green]←[/] [dim]{_short(ev.data.get('result'))}[/]")
        elif ev.type == EventType.DECISION:
            if "remembered" in ev.data:
                for fact in ev.data["remembered"]:
                    activity.write(f"[magenta]remembered:[/] [dim]{_short(fact)}[/]")
            elif "skills_matched" in ev.data:
                names = ", ".join(ev.data["skills_matched"])
                activity.write(f"[blue]skills:[/] [dim]{names}[/]")
            elif "skill_authored" in ev.data:
                activity.write(f"[magenta]new skill:[/] [dim]{ev.data['skill_authored']}[/]")
            elif "blocked" in ev.data:
                activity.write(
                    f"[red]blocked:[/] [dim]{ev.data['blocked']} "
                    f"({ev.data.get('reason', '')})[/]"
                )
            elif "halted" in ev.data:
                activity.write(f"[red]halted:[/] [dim]{_short(ev.data['halted'])}[/]")
            elif ev.data.get("reason") == "max_steps_reached":
                activity.write("[red]! reached the step limit[/]")
        elif ev.type == EventType.USAGE:
            self._tokens += int(ev.data.get("total_tokens", 0))
            self.query_one("#stats", Static).update(self._stats_text())

    # ---- input ---------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#prompt", Input).value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
            return

        transcript = self.query_one("#transcript", RichLog)
        transcript.write(f"\n[b cyan]you[/] › {text}")
        self.query_one("#activity", RichLog).write(f"[dim]— run: {_short(text, 60)}[/]")
        self.query_one("#prompt", Input).disabled = True
        self._worker = self.run_worker(self._run_agent(text), exclusive=True, name="agent")

    async def _run_agent(self, text: str) -> None:
        transcript = self.query_one("#transcript", RichLog)
        try:
            answer = await self.agent.run(text)
            transcript.write(f"[b green]dae[/] › {answer}")
        except Exception as exc:  # noqa: BLE001 - surface errors, never crash the UI
            transcript.write(f"[b red]error:[/] {exc}")
        finally:
            self._worker = None
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()

    # ---- slash commands ------------------------------------------------------

    def _handle_command(self, text: str) -> None:
        cmd = text.split()[0].lower()
        transcript = self.query_one("#transcript", RichLog)
        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/help":
            transcript.write(_HELP)
        elif cmd == "/clear":
            transcript.clear()
        elif cmd == "/reset":
            self.agent.history.clear()
            transcript.write("[dim]conversation history cleared[/]")
        elif cmd == "/memory":
            self._show_memory(transcript)
        elif cmd == "/skills":
            self._show_skills(transcript)
        elif cmd == "/jobs":
            self._show_jobs(transcript)
        elif cmd == "/budget":
            self._show_budget(transcript)
        elif cmd == "/stop":
            # Hard kill switch: engage the stop-file (halts every run, even other
            # processes) AND cancel anything in flight in this app.
            budget = self.agent.budget
            if budget is not None:
                budget.engage_kill_switch()
            if self._worker is not None:
                self._worker.cancel()
            transcript.write(
                "[bold red]Kill switch engaged.[/] [dim]All runs halt until /resume.[/]"
            )
        elif cmd == "/resume":
            budget = self.agent.budget
            if budget is not None and budget.release_kill_switch():
                transcript.write("[bold green]Kill switch released.[/]")
            else:
                transcript.write("[dim]kill switch was not engaged.[/]")
        else:
            transcript.write(f"[red]unknown command:[/] {cmd}  [dim](/help for the list)[/]")

    def _show_memory(self, transcript: RichLog) -> None:
        memory = self.agent.memory
        if memory is None:
            transcript.write("[dim]memory is not enabled.[/]")
            return
        facts = memory.recent(limit=10)
        if not facts:
            transcript.write("[dim]nothing remembered yet.[/]")
            return
        transcript.write(f"[b]Remembered facts[/] [dim]({memory.count()} total)[/]")
        for fact in facts:
            transcript.write(f"[dim]  - {fact}[/]")

    def _show_skills(self, transcript: RichLog) -> None:
        library = self.agent.skills
        skills = library.all() if library else []
        if not skills:
            transcript.write("[dim]no skills loaded.[/]")
            return
        transcript.write(f"[b]Skills[/] [dim]({len(skills)} available)[/]")
        for skill in skills:
            transcript.write(f"[cyan]  {skill.name}[/] [dim]- {skill.description}[/]")

    def _show_jobs(self, transcript: RichLog) -> None:
        from ..scheduler.jobs import JobStore

        store = JobStore(self.settings.dae_home / "jobs.db")
        try:
            jobs = store.all()
        finally:
            store.close()
        if not jobs:
            transcript.write(
                '[dim]no scheduled jobs. Add one:[/] dae jobs add "every day at 9am" "<prompt>"'
            )
            return
        transcript.write(f"[b]Scheduled jobs[/] [dim]({len(jobs)})[/]")
        for job in jobs:
            state = "[green]on[/]" if job.enabled else "[yellow]paused[/]"
            transcript.write(
                f"[cyan]  {job.id}[/] {state} [dim]{job.human}[/] - {_short(job.prompt, 60)}"
            )

    def _show_budget(self, transcript: RichLog) -> None:
        budget = self.agent.budget
        if budget is None:
            transcript.write("[dim]budget tracking is not enabled.[/]")
            return
        snap = budget.snapshot()
        cap = float(snap["daily_cap_usd"])
        transcript.write("[b]Budget[/]")
        if cap > 0:
            transcript.write(
                f"[dim]  today: ${snap['daily_cost_usd']:.4f} / ${cap:.2f} cap "
                f"(remaining ${snap['remaining_usd']:.4f})[/]"
            )
        else:
            transcript.write(f"[dim]  today: ${snap['daily_cost_usd']:.4f} spent / no daily cap[/]")
        transcript.write(
            f"[dim]  session: {snap['session_tokens']} tokens / ${snap['session_cost_usd']:.4f}[/]"
        )
        transcript.write(f"[dim]  kill switch: {'ENGAGED' if snap['killed'] else 'off'}[/]")

    # ---- helpers -------------------------------------------------------------

    def _stats_text(self) -> str:
        return (
            f"[b]provider[/] {self.settings.model_provider}\n"
            f"[b]model[/] {self.settings.model_name}\n"
            f"[b]tokens[/] {self._tokens}"
        )


def run_tui(agent: Agent, settings: Settings) -> None:
    """Launch the TUI for an already-constructed agent. Caller owns cleanup of the
    agent's provider and memory store (see :func:`daedalus.cli._run_tui`)."""
    DaedalusApp(agent, settings).run()
