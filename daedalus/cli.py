"""``dae`` entrypoint.

By default ``dae`` opens the full-screen Textual TUI (:mod:`daedalus.surfaces.tui`);
``dae --plain`` keeps the minimal line-based REPL below. Both build the agent the
same way via :func:`_build_agent` and both lean on the event bus for live
observability — the REPL prints a dim token tally and tool activity per turn, proof
that the agent's nervous system is wired from day one.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from . import __version__
from .config import get_settings
from .core.curl import parse_curl
from .core.events import Event, EventType, get_event_bus
from .core.llm import LLMProvider, OpenAICompatibleProvider, build_provider
from .core.loop import Agent
from .memory.files import ensure_seeded
from .memory.graph import MemoryGraph
from .memory.semantic import SemanticIndex
from .memory.store import MemoryStore
from .reflection.critic import Critic
from .safety import ApprovalRequest, AuditLog, BudgetGovernor, Guardrails
from .safety.guardrails import Approver
from .skills.engine import build_library
from .subagents.spawn import SubagentSpawner
from .tools.mcp_client import MCPManager
from .tools.registry import build_registry

console = Console()

_HELP = """\
Commands:
  /help     show this help
  /memory   show what Daedalus remembers
  /skills   list the skills (playbooks) Daedalus can use
  /jobs     list your scheduled tasks (manage them with `dae jobs ...`)
  /budget   show today's spend, session tokens, and the kill-switch state
  /stop     engage the kill switch (halts every run until /resume)
  /resume   release the kill switch
  /reset    clear the conversation history
  /exit     quit (also Ctrl-D / Ctrl-C)
Anything else is sent to the agent.\
"""


@dataclass(slots=True)
class AgentSession:
    """Everything one surface needs for a chat session, built by :func:`_build_agent`.

    Teardown is split in two because the HTTP client closes asynchronously: call
    :meth:`close` for the synchronous resources (audit subscription + memory store),
    then ``await session.provider.aclose()``. The agent carries its own ``budget`` and
    ``guardrails`` (``session.agent.budget`` / ``.guardrails``), so surfaces reach the
    safety rails through it without extra plumbing.
    """

    agent: Agent
    memory: MemoryStore
    provider: LLMProvider
    detach_audit: Callable[[], None]
    mcp: MCPManager | None = None
    graph: MemoryGraph | None = None
    semantic: SemanticIndex | None = None

    def close(self) -> None:
        """Release the synchronous resources (best-effort, in reverse order)."""
        self.detach_audit()
        if self.mcp is not None:
            self.mcp.stop()
        if self.semantic is not None:
            self.semantic.close()
        if self.graph is not None:
            self.graph.close()
        self.memory.close()


def _build_agent(settings, approver: Approver | None = None) -> AgentSession:
    """Construct the agent and the resources it owns (provider, memory, safety rails).

    Shared by both surfaces — the plain REPL and the Textual TUI — so they wire up
    tools, memory, skills, and safety identically. The optional ``approver`` is how a
    surface answers an ``ask``-policy approval prompt; pass ``None`` for headless runs
    (where ``ask`` fails closed) or set ``agent.guardrails.approver`` later.
    """
    provider = build_provider(settings)
    # Memory: seed the human-editable Markdown files on first run, then open the
    # SQLite store the loop archives turns and recalls facts from.
    ensure_seeded()
    memory = MemoryStore(settings.dae_home / "state.db")
    # Skills: load the playbook library (seeded on first run); the agent matches and
    # injects relevant ones per turn and can author new ones after successful tasks.
    skills = build_library()
    registry = build_registry()

    # MCP: connect to any configured Model Context Protocol servers and graft their
    # tools into the registry (namespaced <server>__<tool>). No-op with zero import cost
    # when none are configured or the [mcp] extra isn't installed. Registered before the
    # guardrails so a server can mark its tools requires_approval and have it enforced.
    mcp = MCPManager(settings.mcp_servers)
    mcp_tools = mcp.start()
    if mcp_tools:
        registry.register(mcp_tools)

    # M10 memory upgrades, both sharing the one state.db (so there's a single file to
    # back up). The graph is $0 and on by default; semantic recall needs the [semantic]
    # extra + opt-in and is otherwise a transparent no-op.
    graph = MemoryGraph(settings.dae_home / "state.db") if settings.memory_graph else None
    semantic = SemanticIndex(settings.dae_home / "state.db", enabled=settings.semantic_memory)

    # Subagents: register the depth/count-capped spawn tool so the model can delegate
    # focused subtasks. The spawner shares the budget + guardrails it's given below.
    spawner: SubagentSpawner | None = None

    # Safety rails, all wired into the one loop (see daedalus.safety):
    #  - budget: per-day spend cap + session totals + the kill switch (a stop-file).
    #  - guardrails: approval policy for tools that can change the world.
    #  - audit: an append-only JSONL of every bus event, attached as a subscriber.
    budget = BudgetGovernor(settings)
    guardrails = Guardrails(registry, policy=settings.approval_policy, approver=approver)
    audit = AuditLog(settings.dae_home / "audit.log")
    detach_audit = audit.attach(get_event_bus())

    # Now that budget + guardrails exist, build the subagent spawner (it shares both so
    # child agents obey the same cap and approval gate) and register its tool.
    if settings.subagents:
        spawner = SubagentSpawner(
            provider,
            settings,
            registry,
            budget=budget,
            guardrails=guardrails,
            max_depth=settings.subagent_max_depth,
            max_children=settings.subagent_max_children,
        )
        registry.register([spawner.tool(depth=0)])

    # Reflection critic: an extra self-review pass on each answer when REFLECTION=true.
    critic = (
        Critic(provider, max_revisions=settings.reflection_max_revisions)
        if settings.reflection
        else None
    )

    agent = Agent(
        provider,
        settings,
        tools=registry,
        memory=memory,
        skills=skills,
        budget=budget,
        guardrails=guardrails,
        graph=graph,
        semantic=semantic,
        critic=critic,
        subagents=spawner,
    )
    return AgentSession(
        agent, memory, provider, detach_audit, mcp=mcp, graph=graph, semantic=semantic
    )


def _short(value: object, limit: int = 90) -> str:
    """Collapse whitespace and clip a value for one-line console output."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _print_budget(budget: BudgetGovernor) -> None:
    """Render the budget snapshot for the ``/budget`` command."""
    snap = budget.snapshot()
    cap = float(snap["daily_cap_usd"])
    console.print("[bold]Budget[/]")
    if cap > 0:
        console.print(
            f"[dim]  today:   ${snap['daily_cost_usd']:.4f} spent / ${cap:.2f} cap "
            f"(remaining ${snap['remaining_usd']:.4f})[/]"
        )
    else:
        console.print(f"[dim]  today:   ${snap['daily_cost_usd']:.4f} spent / no daily cap[/]")
    console.print(
        f"[dim]  session: {snap['session_tokens']} tokens / ${snap['session_cost_usd']:.4f}[/]"
    )
    console.print(f"[dim]  kill switch: {'ENGAGED' if snap['killed'] else 'off'}[/]")


async def _run_repl() -> None:
    settings = get_settings()

    # The "thinking" spinner is a Rich Live display; we pause it while reading an
    # approval answer so the y/N prompt isn't trampled by spinner repaints. The holder
    # lets the approver (built before the agent) reach the status created per turn.
    current_status: dict[str, object] = {"obj": None}

    async def approver(req: ApprovalRequest) -> bool:
        status = current_status.get("obj")
        if status is not None:
            status.stop()
        try:
            console.print(
                f"\n[bold yellow]Approval needed[/] for [bold]{req.tool}[/] "
                f"[dim]({req.reason})[/]"
            )
            console.print(f"[dim]  args: {_short(req.args, 200)}[/]")
            try:
                return console.input("  allow this? [y/N] ").strip().lower() in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return False
        finally:
            if status is not None:
                status.start()

    session = _build_agent(settings, approver=approver)
    agent, memory, budget = session.agent, session.memory, session.agent.budget

    # Subscribe to the bus to tally tokens and show tool activity live.
    turn_tokens = {"total": 0}

    def on_event(ev: Event) -> None:
        if ev.type == EventType.USAGE:
            turn_tokens["total"] += int(ev.data.get("total_tokens", 0))
        elif ev.type == EventType.TOOL_CALL:
            console.print(f"[dim]  -> {ev.data.get('name')}({_short(ev.data.get('args'))})[/]")
        elif ev.type == EventType.TOOL_RESULT:
            console.print(f"[dim]  <- {_short(ev.data.get('result'))}[/]")

    unsubscribe = get_event_bus().subscribe(on_event)

    console.print(
        f"[bold cyan]Daedalus[/] v{__version__}  "
        f"[dim]provider={settings.model_provider} model={settings.model_name}[/]"
    )
    console.print("[dim]Type /help for commands.[/]\n")

    try:
        while True:
            try:
                user = console.input("[bold]you[/] > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/]")
                break

            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            if user == "/help":
                console.print(_HELP)
                continue
            if user == "/reset":
                agent.history.clear()
                console.print("[dim]history cleared[/]")
                continue
            if user == "/memory":
                facts = memory.recent(limit=10)
                if facts:
                    console.print(f"[bold]Remembered facts[/] [dim]({memory.count()} total)[/]")
                    for fact in facts:
                        console.print(f"[dim]  - {fact}[/]")
                else:
                    console.print("[dim]nothing remembered yet[/]")
                console.print(
                    f"[dim]  edit your persona & memory by hand: {settings.dae_home}"
                    "  (SOUL.md, MEMORY.md, USER.md)[/]"
                )
                continue
            if user == "/skills":
                skills = agent.skills.all() if agent.skills else []
                if skills:
                    console.print(f"[bold]Skills[/] [dim]({len(skills)} available)[/]")
                    for skill in skills:
                        console.print(f"[cyan]  {skill.name}[/] [dim]- {skill.description}[/]")
                else:
                    console.print("[dim]no skills loaded[/]")
                continue
            if user == "/jobs":
                _print_jobs(settings)
                continue
            if user == "/budget":
                if budget is not None:
                    _print_budget(budget)
                continue
            if user == "/stop":
                if budget is not None:
                    budget.engage_kill_switch()
                    console.print(
                        "[bold red]Kill switch engaged.[/] [dim]Runs halt until /resume.[/]"
                    )
                continue
            if user == "/resume":
                if budget is not None:
                    existed = budget.release_kill_switch()
                    console.print(
                        "[bold green]Kill switch released.[/]"
                        if existed
                        else "[dim]Kill switch was not engaged.[/]"
                    )
                continue

            turn_tokens["total"] = 0
            status = console.status("[dim]thinking...[/]", spinner="dots")
            current_status["obj"] = status
            status.start()
            try:
                answer = await agent.run(user)
            except Exception as exc:  # noqa: BLE001 - show errors, never crash the REPL
                console.print(f"[bold red]error:[/] {exc}")
                continue
            finally:
                status.stop()
                current_status["obj"] = None

            console.print(f"[bold green]dae[/] > {answer}")
            if turn_tokens["total"]:
                console.print(f"[dim]  - {turn_tokens['total']} tokens[/]")
    finally:
        unsubscribe()
        session.close()
        await session.provider.aclose()


def _run_tui() -> None:
    """Launch the Textual TUI (the default surface). Textual is imported lazily so the
    lighter ``--plain`` REPL and ``--help`` don't pay for it.

    The agent is built without an approver here; the TUI installs its own modal
    approver on ``agent.guardrails`` once the app is running (see :mod:`surfaces.tui`).
    """
    from .surfaces.tui import run_tui

    settings = get_settings()
    session = _build_agent(settings)
    try:
        run_tui(session.agent, settings)
    finally:
        session.close()
        asyncio.run(session.provider.aclose())


def _run_web(host: str, port: int) -> None:
    """Serve the web chat UI + Trace Viewer. Needs the ``[web]`` extra (FastAPI/uvicorn).

    The agent installs no approver of its own here; the web connection sets a
    per-socket approver on ``agent.guardrails`` for the duration of each run (see
    :mod:`surfaces.web`), so destructive tools raise an Allow/Deny card in the browser.
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[bold red]The web surface needs the optional 'web' extra.[/]\n"
            '[dim]Install it with:[/]  uv pip install -e ".[web]"'
        )
        raise SystemExit(1) from None
    from .surfaces.web import create_app

    settings = get_settings()
    session = _build_agent(settings)
    app = create_app(session.agent, settings, enable_scheduler=True)
    console.print(
        f"[bold cyan]Daedalus[/] web UI at [b]http://{host}:{port}[/]  "
        f"[dim]({settings.model_provider}/{settings.model_name}, Ctrl-C to stop)[/]"
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        session.close()
        asyncio.run(session.provider.aclose())


def _run_telegram() -> None:
    """Run the Telegram bot gateway. Needs the ``[telegram]`` extra and a bot token.

    The bot installs its own inline-keyboard approver on ``agent.guardrails`` and, if the
    ``[scheduler]`` extra is present, runs user-defined jobs and texts you the results.
    """
    try:
        from .surfaces.telegram_bot import run_telegram
    except ImportError:
        console.print(
            "[bold red]The Telegram surface needs the optional 'telegram' extra.[/]\n"
            '[dim]Install it with:[/]  uv pip install -e ".[telegram]"'
        )
        raise SystemExit(1) from None

    settings = get_settings()
    session = _build_agent(settings)
    try:
        run_telegram(session.agent, settings)
    finally:
        session.close()
        asyncio.run(session.provider.aclose())


def _run_whatsapp() -> None:
    """Run the (unofficial) WhatsApp bridge. Needs Node.js; see bridge/whatsapp/README.md."""
    from .surfaces.whatsapp import run_whatsapp

    settings = get_settings()
    session = _build_agent(settings)
    try:
        run_whatsapp(session.agent, settings)
    finally:
        session.close()
        asyncio.run(session.provider.aclose())


def _run_voice(once_file: str | None = None) -> None:
    """Talk to Daedalus by voice (needs the ``[voice]`` extra; mic loop needs sounddevice).

    Default: a push-to-talk loop — record a few seconds, transcribe, answer, speak the
    reply. ``--file FOO.wav`` runs one turn from an existing recording (handy for testing
    without a microphone). Every dependency failure is reported with its exact install hint.
    """
    from .voice.io import VoiceIO, VoiceUnavailable

    settings = get_settings()
    session = _build_agent(settings)
    voice = VoiceIO(settings)
    workspace = settings.ensure_home() / "voice"
    workspace.mkdir(parents=True, exist_ok=True)

    async def one_turn(wav_in: Path) -> None:
        # Transcription / playback are blocking (CPU model, audio device), so run them
        # off the event loop to keep it responsive.
        heard = await asyncio.to_thread(voice.transcribe, wav_in)
        if not heard:
            console.print("[yellow]Heard nothing.[/]")
            return
        console.print(f"[bold cyan]You:[/] {heard}")
        answer = await session.agent.run(heard)
        console.print(f"[bold green]Daedalus:[/] {answer}")
        try:
            out = await asyncio.to_thread(voice.speak_to_file, answer, workspace / "reply.wav")
            await asyncio.to_thread(voice.play, out)
        except VoiceUnavailable as exc:
            console.print(f"[dim](no speech output: {exc})[/]")

    async def session_loop() -> None:
        if once_file:
            await one_turn(Path(once_file))
            return
        console.print("[bold cyan]Voice mode.[/] [dim]Recording ~5s per turn; Ctrl-C to stop.[/]")
        while True:
            console.print("[dim]listening…[/]")
            try:
                clip = await asyncio.to_thread(voice.record, workspace / "in.wav", 5.0)
            except VoiceUnavailable as exc:
                console.print(f"[bold red]{exc}[/]")
                return
            await one_turn(clip)

    async def main_async() -> None:
        try:
            await session_loop()
        finally:
            session.close()
            await session.provider.aclose()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[dim]voice mode stopped.[/]")


# ---- `dae jobs`: manage user-defined scheduled tasks ------------------------


def _print_jobs(settings) -> None:
    """Render the scheduled-job list for the ``/jobs`` command and ``dae jobs list``."""
    from .scheduler.jobs import JobStore

    store = JobStore(settings.dae_home / "jobs.db")
    try:
        jobs = store.all()
    finally:
        store.close()
    if not jobs:
        console.print(
            '[dim]No scheduled jobs. Add one: [/]dae jobs add "every day at 9am" "<prompt>"'
        )
        return
    console.print(f"[bold]Scheduled jobs[/] [dim]({len(jobs)})[/]")
    for job in jobs:
        state = "[green]on[/]" if job.enabled else "[yellow]paused[/]"
        console.print(f"  [cyan]{job.id}[/] {state} [dim]{job.human}[/] — {_short(job.prompt, 70)}")


def _run_jobs(args: argparse.Namespace) -> int:
    """CRUD for scheduled jobs, operating directly on the persistent store.

    No agent or running server is required to manage jobs — the next time a long-running
    surface (``dae serve`` / ``dae telegram``) starts, it picks them up from the store.
    """
    from .scheduler.jobs import JobStore, ScheduleError

    settings = get_settings()
    if args.jobs_command in (None, "list"):
        _print_jobs(settings)
        return 0

    store = JobStore(settings.dae_home / "jobs.db")
    try:
        if args.jobs_command == "add":
            try:
                job = store.add(args.prompt, args.schedule)
            except ScheduleError as exc:
                console.print(f"[bold red]{exc}[/]")
                return 1
            console.print(
                f"[bold green]Added job[/] [cyan]{job.id}[/] [dim]({job.human})[/]\n"
                "[dim]It runs when `dae serve` or `dae telegram` is up.[/]"
            )
            return 0
        if args.jobs_command in ("rm", "remove", "delete"):
            ok = store.delete(args.id)
            console.print(
                f"[green]Deleted {args.id}.[/]" if ok else f"[yellow]No job {args.id}.[/]"
            )
            return 0 if ok else 1
        if args.jobs_command in ("pause", "resume"):
            ok = store.set_enabled(args.id, args.jobs_command == "resume")
            verb = "Resumed" if args.jobs_command == "resume" else "Paused"
            console.print(f"[green]{verb} {args.id}.[/]" if ok else f"[yellow]No job {args.id}.[/]")
            return 0 if ok else 1
    finally:
        store.close()
    return 0


# ---- `dae connect`: wire up a custom endpoint from a pasted curl ------------


def _mask(secret: str | None) -> str:
    """Show only the last 4 chars of a key so confirmations don't leak it."""
    if not secret:
        return "(none)"
    return "..." + secret[-4:] if len(secret) > 4 else "****"


def _write_env(updates: dict[str, str], path: Path) -> None:
    """Upsert ``KEY=value`` lines into a .env file, preserving everything else.

    Existing keys are rewritten in place; new keys are appended. We never touch
    comments or unrelated settings, so a hand-edited .env survives intact.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, value in remaining.items():  # keys not already present
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


async def _run_connect(curl_text: str, do_test: bool, env_path: Path) -> int:
    """Parse a curl command, persist a clean custom-provider config, optionally test."""
    cfg = parse_curl(curl_text)
    if not cfg.base_url:
        console.print(
            "[bold red]Could not find a URL in that curl command.[/]\n"
            "[dim]Paste the full command, including the https:// endpoint.[/]"
        )
        return 1

    console.print("[bold]Parsed from curl:[/]")
    console.print(f"  base_url  [cyan]{cfg.base_url}[/]")
    console.print(f"  model     [cyan]{cfg.model or '(not found - keep current MODEL_NAME)'}[/]")
    console.print(f"  api_key   [cyan]{_mask(cfg.api_key)}[/]")

    if do_test:
        console.print("\n[dim]Testing the connection...[/]")
        provider = OpenAICompatibleProvider(
            base_url=cfg.base_url,
            model=cfg.model or "gpt-3.5-turbo",
            api_key=cfg.api_key,
            timeout=get_settings().request_timeout,
        )
        try:
            with console.status("[dim]calling the endpoint...[/]", spinner="dots"):
                resp = await provider.chat([{"role": "user", "content": "Reply with the word: ok"}])
            console.print(
                f"[bold green]Connected.[/] [dim]Model said: {(resp.text or '').strip()[:60]}[/]"
            )
        except Exception as exc:  # noqa: BLE001 - show the failure, don't crash
            console.print(f"[bold red]Test failed:[/] {exc}")
            console.print("[dim]Saving the config anyway — fix the endpoint/key and retry.[/]")
        finally:
            await provider.aclose()

    updates = {"MODEL_PROVIDER": "custom", "OPENAI_BASE_URL": cfg.base_url}
    if cfg.api_key:
        updates["OPENAI_API_KEY"] = cfg.api_key
    if cfg.model:
        updates["MODEL_NAME"] = cfg.model
    _write_env(updates, env_path)
    console.print(f"\n[bold green]Saved to {env_path}.[/] Run [bold]dae[/] to start chatting.")
    return 0


def _read_curl_arg(curl: str | None) -> str:
    """Get the curl text from the argument, or read it from stdin if piped."""
    if curl:
        return curl
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print(
        "[bold]Paste your curl command, then press Enter:[/]\n"
        "[dim](or run: dae connect 'curl https://HOST/v1/chat/completions "
        '-H "Authorization: Bearer KEY" -d ...\')[/]'
    )
    try:
        return console.input("> ")
    except (EOFError, KeyboardInterrupt):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(prog="dae", description="Daedalus - a self-hosted AI agent.")
    parser.add_argument("--version", action="version", version=f"dae {__version__}")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="use the minimal line-based REPL instead of the full-screen TUI",
    )
    sub = parser.add_subparsers(dest="command")

    connect = sub.add_parser(
        "connect",
        help="connect to any OpenAI-compatible endpoint by pasting its curl command",
    )
    connect.add_argument("curl", nargs="?", help="the curl command (reads stdin if omitted)")
    connect.add_argument(
        "--no-test", action="store_true", help="skip the live connection test before saving"
    )

    serve = sub.add_parser(
        "serve",
        aliases=["web"],
        help="serve the web chat UI + Trace Viewer (needs the 'web' extra)",
    )
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port (default 8000)")

    sub.add_parser("telegram", help="run the Telegram bot gateway (needs the 'telegram' extra)")
    sub.add_parser("whatsapp", help="run the unofficial WhatsApp bridge (needs Node.js)")

    voice = sub.add_parser("voice", help="talk to Daedalus by voice (needs the 'voice' extra)")
    voice.add_argument(
        "--file", dest="voice_file", help="run one turn from a WAV file instead of the mic"
    )

    jobs = sub.add_parser("jobs", help="manage user-defined scheduled tasks")
    jobs_sub = jobs.add_subparsers(dest="jobs_command")
    jobs_sub.add_parser("list", help="list all scheduled jobs (default)")
    jobs_add = jobs_sub.add_parser("add", help='add a job: dae jobs add "<when>" "<prompt>"')
    jobs_add.add_argument("schedule", help="natural-language schedule, e.g. 'every day at 9am'")
    jobs_add.add_argument("prompt", help="what to ask Daedalus to do each time")
    for verb in ("rm", "pause", "resume"):
        p = jobs_sub.add_parser(verb, help=f"{verb} a job by id")
        p.add_argument("id", help="the job id (see `dae jobs list`)")

    args = parser.parse_args()

    if args.command == "connect":
        curl_text = _read_curl_arg(args.curl)
        if not curl_text.strip():
            console.print("[red]No curl command provided.[/]")
            raise SystemExit(1)
        raise SystemExit(asyncio.run(_run_connect(curl_text, not args.no_test, Path(".env"))))

    if args.command in ("serve", "web"):
        _run_web(args.host, args.port)
        return

    if args.command == "telegram":
        _run_telegram()
        return

    if args.command == "whatsapp":
        _run_whatsapp()
        return

    if args.command == "voice":
        _run_voice(args.voice_file)
        return

    if args.command == "jobs":
        raise SystemExit(_run_jobs(args))

    if args.plain:
        asyncio.run(_run_repl())
    else:
        _run_tui()


if __name__ == "__main__":
    main()
