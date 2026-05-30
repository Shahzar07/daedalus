# Daedalus (`dae`)

A self-hosted, **teachable** AI agent that remembers across sessions, writes its own
reusable skills from experience, and reaches you on every surface — terminal, web, Telegram,
and WhatsApp. Runs **$0** on a laptop with local models, and scales up to any hosted provider
you plug in. The agent loop is hand-rolled and readable on purpose: this is a learning
artifact, not a framework black box.

> **Status: complete (M1–M11).** The hand-rolled ReAct loop, pluggable tools, cross-session
> memory, self-authored skills, the safety layer (sandbox + guardrails + budget + kill
> switch), the web Trace Viewer, the Telegram/WhatsApp gateways, the natural-language
> scheduler, the MCP client, and the M10 upgrades (reflection, subagents, memory graph,
> semantic memory, voice) all run today. The whole stack works offline and $0 through the
> built-in `mock` provider — which is how the **173-test** suite stays green with no network
> and no keys.

## Why Daedalus

- **Remembers you.** SQLite/FTS5 recall + an optional semantic index + a lightweight memory
  graph, plus hand-editable `SOUL.md` / `MEMORY.md` / `USER.md` — all on your machine.
- **Writes its own skills.** After solving a multi-step task it distills a reusable
  `SKILL.md` playbook, then matches and reuses it next time.
- **Everywhere you are.** Start a task in the terminal, continue it from your phone over
  Telegram or WhatsApp, or watch it think in the browser.
- **Yours, and safe.** Local-first, no telemetry, no phone-home. Shell/file actions are
  sandboxed, destructive ones need approval, and a hard budget cap + `/stop` kill switch
  bound every run — including scheduled ones.
- **Legible.** Every thought, tool call, and decision is an event you can watch live or
  replay in the Trace Viewer.

## Quickstart

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install — installs `dae` as a global command, seeds .env, prepares a dev venv
./install.sh           # macOS / Linux
./install.ps1          # Windows (PowerShell)

# 2. Run — open a new terminal so `dae` is on PATH, then just:
dae                    # first run opens a quick setup wizard, then your chosen surface
```

On that **first run** Daedalus walks you through a two-step setup — pick a model provider
(and drop in a key, or paste a provider's `curl`), then pick how you want to talk to it
(terminal UI, web, Telegram, WhatsApp). It writes your `.env` for you and launches. After
that:

```bash
dae                    # full-screen streaming terminal UI (default)
dae setup              # re-run the wizard anytime (re-pick provider / surface)
dae --plain            # minimal line REPL
dae --help             # every command
```

`dae` works like any normal CLI on PowerShell, bash, or zsh. (Prefer not to install
globally? `uv run dae` runs it straight from the project venv instead.) To give the
global command optional powers, install it with extras:
`uv tool install --editable ".[web,telegram,voice,scheduler,semantic,mcp]"`.

**Zero-key path:** install [Ollama](https://ollama.com), `ollama pull qwen2.5`, and Daedalus
talks to it out of the box. **Hosted path:** set `MODEL_PROVIDER` + the matching key in `.env`
(Groq, Gemini, OpenRouter, OpenAI). **Any other endpoint:** paste its `curl` and Daedalus
wires itself up — `dae connect 'curl https://api.example.com/v1/chat/completions -H "Authorization: Bearer KEY" -d "{\"model\":\"NAME\"}"'`.

**No model at all?** The built-in `mock` provider runs the whole stack — loop, tools, memory,
UI — offline and free:

```bash
MODEL_PROVIDER=mock dae             # try the UI with scripted responses (bash/zsh)
uv run pytest                       # 173 tests, $0, no network, no keys
```

> On PowerShell, set it inline: `$env:MODEL_PROVIDER='mock'; dae`

## What it can do — by command

| Command | What it does | Needs |
|---|---|---|
| `dae` | Full-screen Textual TUI: streaming output, live tool panel, slash commands | core |
| `dae --plain` | Minimal line REPL (same agent, no TUI) | core |
| `dae connect '<curl>'` | Connect any OpenAI-compatible endpoint by pasting its curl; live-tests + writes `.env` | core |
| `dae setup` / `dae init` | Re-run the first-run wizard: pick a provider (+ key or curl) and a surface; writes `.env` | core |
| `dae serve` / `dae web` | Web chat UI **+ Trace Viewer** at `http://127.0.0.1:8000` | `[web]` |
| `dae telegram` | Telegram bot gateway (allow-list via `TELEGRAM_ALLOWED_CHAT_IDS`) | `[telegram]` |
| `dae whatsapp` | Unofficial WhatsApp bridge (QR login, local Node companion) | Node.js |
| `dae voice` | Talk by mic: Whisper STT → agent → Piper TTS. `--file x.wav` for one turn | `[voice]` |
| `dae jobs add "every day at 9am" "<prompt>"` | Schedule a **user-defined** recurring task | `[scheduler]` |
| `dae jobs list / pause / resume / rm <id>` | Manage scheduled jobs (persist in `~/.dae/jobs.db`) | `[scheduler]` |

**Slash commands** (TUI and `--plain`): `/help` · `/memory` · `/skills` · `/jobs` · `/budget`
· `/stop` (kill switch) · `/resume` · `/reset` · `/exit`.

## Architecture

```
You ─▶ surface:  TUI · Web · Telegram · WhatsApp · Voice · Scheduler
          │            (every surface builds the SAME Agent via _build_agent)
          ▼
   core/loop.py  ── the request-scoped ReAct loop ─────────────────┐
     │  1. assemble context  (context.py)                          │
     │       SOUL.md persona + USER.md + MEMORY.md + recall + skills│
     │  2. call model        (llm.py: any provider, one interface) │  every step
     │  3. parse → final answer | tool calls                       │  emits an Event
     │  4. execute tools     (tools/registry.py, guardrails gate)  │   ─▶ events.py
     │  5. remember (memory/) + maybe author a skill (skills/)     │   ─▶ audit log
     │  6. optional reflection pass (reflection/critic.py)         │   ─▶ Trace Viewer
     └────────── repeat until final answer or MAX_STEPS ───────────┘
          │
          ├─ tools/        web_search · files · shell(sandbox) · spawn_subagent · MCP servers
          ├─ memory/       SQLite+FTS5 · semantic index · memory graph · *.md files
          ├─ safety/       budget cap · daily kill switch · approval guardrails · audit JSONL
          └─ skills/       match → inject playbook → run → auto-author new SKILL.md
```

The loop is **request-scoped**: it starts when you send a message and ends when it returns an
answer. It never loops on its own and invents no goals. Scheduled jobs are simply stored
user instructions, run later through this exact loop under the same budget + guardrails.

## Providers

Set `MODEL_PROVIDER` in `.env`. Every provider speaks the OpenAI-compatible HTTP API, so one
lean client covers most of them — no heavy SDKs.

| Provider | Cost | Key | Notes |
|---|---|---|---|
| `ollama` | $0 local | none | **default**; `ollama pull qwen2.5` |
| `groq` | free tier | `GROQ_API_KEY` | very fast |
| `gemini` | free tier | `GEMINI_API_KEY` | Google, vision-capable |
| `openrouter` | free + paid | `OPENROUTER_API_KEY` | many models incl. `:free` |
| `openai` | paid | `OPENAI_API_KEY` | |
| `custom` | any | `OPENAI_BASE_URL` + key | LM Studio, vLLM, a proxy… or just `dae connect` |
| `mock` | $0 offline | none | deterministic, for tests/demos |

## Optional extras

The core install stays lean and $0. Heavier capabilities are opt-in:

```bash
uv pip install -e ".[web]"        # web UI + Trace Viewer
uv pip install -e ".[telegram]"   # Telegram bot
uv pip install -e ".[voice]"      # faster-whisper STT + Piper TTS + mic loop
uv pip install -e ".[semantic]"   # sentence-transformers recall-by-meaning
uv pip install -e ".[scheduler]"  # APScheduler for `dae jobs`
uv pip install -e ".[mcp]"        # Model Context Protocol client
uv pip install -e ".[browser]"    # Playwright browsing
uv pip install -e ".[dev]"        # pytest, ruff, black
```

## Configuration

Every option is documented in [`.env.example`](.env.example) with safe defaults. The
first-run wizard (`dae setup`) writes your `.env` for you — a project-local `.env` when you
run from the repo, otherwise the global `~/.dae/.env` that the installed `dae` command reads
from any directory. Runtime state lives in `~/.dae/` (override with `DAE_HOME`):

- `SOUL.md` — the agent's persona + operating principles. **Edit it to retune its voice;**
  it takes effect on the next turn.
- `MEMORY.md` — durable world facts (the agent appends here as it learns).
- `USER.md` — your profile and preferences.
- `state.db` — SQLite: archived turns, FTS5 recall, semantic vectors, memory graph.
- `jobs.db` — scheduled jobs. `audit.jsonl` — append-only action log.

Secrets live only in `.env` (gitignored) — **never hardcode keys**.

## Safety model

- **Sandbox.** The `shell` tool runs in a subprocess with a timeout and a scoped working
  directory (Docker isolation when available). File tools are path-scoped to the workspace.
- **Guardrails.** Tools flagged `requires_approval` (and any MCP server you mark) pass through
  an Allow / Ask / Deny gate before running.
- **Budget + kill switch.** A daily USD cap halts runs gracefully on breach; `/stop` engages a
  hard kill switch (a stop-file the loop polls) that pauses everything until `/resume`.
- **Auditable.** Every event is appended to `~/.dae/audit.jsonl` and replayable in the Trace
  Viewer.

## Tests

```bash
uv run pytest          # 173 tests, hermetic ($0, no network, no keys)
uv run ruff check .    # lint
uv run black .         # format
```

The suite uses the `mock` provider and `importorskip` for optional deps, so it stays green on
a bare machine — no Ollama, Docker, or API keys required.

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| 1 | Skeleton + ReAct loop + providers + CLI | ✅ |
| 2 | Tool system + registry + 3 tools | ✅ |
| 3 | Memory (SQLite/FTS5 + SOUL/MEMORY/USER.md) | ✅ |
| 4 | Textual TUI | ✅ |
| 5 | Skills engine + auto-author + library | ✅ |
| 6 | Safety (sandbox, guardrails, budget, kill switch, audit) | ✅ |
| 7 | Web host + Trace Viewer | ✅ |
| 8 | Telegram + WhatsApp + natural-language scheduler | ✅ |
| 9 | MCP client | ✅ |
| 10 | Reflection · subagents · memory graph · semantic memory · voice | ✅ |
| 11 | Polish: installers, docs, seed persona/memory | ✅ |

## License

MIT. Architecture inspired by Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent) (also MIT).
