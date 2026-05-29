"""The web surface — a FastAPI chat server plus the Trace Viewer.

Two ways in:

  * ``GET /``       a minimal vanilla-JS chat page that talks to ``/ws`` over a
                    WebSocket, streaming the agent's thoughts, tool calls, and results
                    as they happen — and raising an inline Allow/Deny card when a
                    world-changing tool needs approval.
  * ``GET /trace``  the Trace Viewer: every past run rendered as an expandable timeline
                    with per-step token and cost counters, read from the append-only
                    audit log (``~/.dae/audit.log``).

Like every other surface this one is just a **bus subscriber** — it never reaches into
the loop. The same :class:`~daedalus.core.loop.Agent` that powers the terminal powers
the web, so memory, skills, and safety behave identically here.

FastAPI/uvicorn live behind the optional ``[web]`` extra; this module is imported lazily
by ``dae serve`` so the core install stays slim.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import Settings
from ..core.events import Event, EventType, get_event_bus
from ..core.loop import Agent
from ..safety import ApprovalRequest

# Event types worth streaming to the browser as the agent works.
_LIVE_TYPES = {
    EventType.THOUGHT,
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.DECISION,
    EventType.USAGE,
}


def _event_dict(ev: Event) -> dict[str, Any]:
    return {"type": str(ev.type), "data": ev.data, "step": ev.step}


def create_app(agent: Agent, settings: Settings, *, enable_scheduler: bool = False) -> FastAPI:
    """Build the FastAPI app around an already-constructed agent.

    Runs are serialized by a single lock: this is a personal, self-hosted server, so
    one shared agent + history is the right model, and the lock keeps two browser tabs
    from interleaving turns into the same conversation.

    When ``enable_scheduler`` is set (``dae serve`` does this) and the ``[scheduler]``
    extra is installed, user-defined jobs run on this server's event loop, sharing the
    same lock so a scheduled task never interleaves with a live chat. Their output lands
    in the audit log, so it shows up in the Trace Viewer like any other run.
    """
    app = FastAPI(title="Daedalus")
    run_lock = asyncio.Lock()
    audit_path = settings.dae_home / "audit.log"
    jobs_path = settings.dae_home / "jobs.db"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _CHAT_HTML

    @app.get("/trace", response_class=HTMLResponse)
    async def trace_page() -> str:
        return _TRACE_HTML

    @app.get("/api/trace")
    async def trace_data() -> JSONResponse:
        return JSONResponse(_read_runs(audit_path))

    @app.get("/api/info")
    async def info() -> JSONResponse:
        return JSONResponse({"provider": settings.model_provider, "model": settings.model_name})

    @app.get("/api/jobs")
    async def jobs_data() -> JSONResponse:
        return JSONResponse({"jobs": _read_jobs(jobs_path)})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await _Connection(websocket, agent, run_lock).serve()

    if enable_scheduler:
        _attach_scheduler(app, agent, settings, run_lock, jobs_path)

    return app


def _attach_scheduler(
    app: FastAPI, agent: Agent, settings: Settings, run_lock: asyncio.Lock, jobs_path: Path
) -> None:
    """Start/stop the scheduler with the app lifecycle (no-op if the extra is missing)."""
    try:
        from ..scheduler.heartbeat import Scheduler
        from ..scheduler.jobs import JobStore
    except ImportError:
        return

    store = JobStore(jobs_path)
    scheduler = Scheduler(agent, store, run_lock=run_lock)

    @app.on_event("startup")
    async def _start_scheduler() -> None:
        await scheduler.start()

    @app.on_event("shutdown")
    async def _stop_scheduler() -> None:
        await scheduler.shutdown()
        store.close()


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    """Read scheduled jobs for ``/api/jobs`` (empty if none / store unavailable)."""
    try:
        from ..scheduler.jobs import JobStore
    except ImportError:
        return []
    if not path.exists():
        return []
    store = JobStore(path)
    try:
        return [
            {
                "id": j.id,
                "human": j.human,
                "prompt": j.prompt,
                "enabled": j.enabled,
                "last_run": j.last_run,
            }
            for j in store.all()
        ]
    finally:
        store.close()


class _Connection:
    """One browser WebSocket: receives messages, drives the agent, streams events."""

    def __init__(self, websocket: WebSocket, agent: Agent, run_lock: asyncio.Lock):
        self.ws = websocket
        self.agent = agent
        self.run_lock = run_lock
        # Approval prompts in flight, keyed by id; resolved when the client answers.
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def serve(self) -> None:
        try:
            while True:
                msg = json.loads(await self.ws.receive_text())
                kind = msg.get("type")
                if kind == "message":
                    await self._handle_message(str(msg.get("text", "")))
                elif kind == "approval_response":
                    fut = self._pending.get(str(msg.get("id")))
                    if fut is not None and not fut.done():
                        fut.set_result(bool(msg.get("allow")))
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a dropped/garbled socket must not crash the server
            return

    async def _approve(self, req: ApprovalRequest) -> bool:
        """Guardrails approver: ask the browser and wait (fails closed on timeout)."""
        approval_id = uuid.uuid4().hex[:8]
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = fut
        try:
            await self.ws.send_json(
                {
                    "type": "approval_request",
                    "id": approval_id,
                    "tool": req.tool,
                    "args": req.args,
                    "reason": req.reason,
                }
            )
            return await asyncio.wait_for(fut, timeout=300)
        except Exception:  # noqa: BLE001 - timeout or send failure => deny
            return False
        finally:
            self._pending.pop(approval_id, None)

    async def _handle_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        run_id = uuid.uuid4().hex[:8]

        async def forward(ev: Event) -> None:
            if ev.run_id == run_id and ev.type in _LIVE_TYPES:
                await self.ws.send_json({"type": "event", "event": _event_dict(ev)})

        unsubscribe = get_event_bus().subscribe(forward)
        # Serialize turns and point the guardrails at this connection for its duration,
        # so an approval prompt during this run lands in this browser tab.
        async with self.run_lock:
            guardrails = self.agent.guardrails
            previous = guardrails.approver if guardrails is not None else None
            if guardrails is not None:
                guardrails.approver = self._approve
            try:
                answer = await self.agent.run(text, run_id=run_id)
                await self.ws.send_json({"type": "final", "text": answer})
            except Exception as exc:  # noqa: BLE001 - surface, don't drop the socket
                await self.ws.send_json({"type": "error", "text": str(exc)})
            finally:
                if guardrails is not None:
                    guardrails.approver = previous
                unsubscribe()


def _read_runs(path: Path, max_runs: int = 100) -> dict[str, Any]:
    """Group the audit log into runs for the Trace Viewer (newest first).

    Reading the whole JSONL each request is fine for a personal tool; we cap the number
    of runs returned so a long-lived log stays responsive.
    """
    if not path.exists():
        return {"runs": []}

    runs: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"runs": []}

    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        rid = str(rec.get("run_id") or "?")
        run = runs.get(rid)
        if run is None:
            run = {"run_id": rid, "events": [], "tokens": 0, "cost_usd": 0.0, "ts": rec.get("ts")}
            runs[rid] = run
            order.append(rid)
        run["events"].append(rec)
        if rec.get("type") == "usage":
            data = rec.get("data") or {}
            run["tokens"] += int(data.get("total_tokens", 0) or 0)
            run["cost_usd"] += float(data.get("cost_usd", 0) or 0.0)

    selected = [runs[r] for r in reversed(order)][:max_runs]
    return {"runs": selected}


# --- embedded front-end (vanilla JS, no build step) --------------------------

_STYLE = """
:root {
  --bg: #0d1117; --panel: #161b22; --border: #30363d; --text: #e6edf3;
  --muted: #8b949e; --accent: #58a6ff; --you: #1f6feb; --dae: #238636;
  --warn: #d29922; --err: #f85149;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
header { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
  border-bottom: 1px solid var(--border); background: var(--panel); }
header h1 { font-size: 16px; margin: 0; letter-spacing: .5px; }
header .meta { color: var(--muted); font-size: 12px; }
header a { color: var(--accent); text-decoration: none; font-size: 13px; margin-left: auto; }
main { flex: 1; display: flex; min-height: 0; }
#chat { flex: 2; display: flex; flex-direction: column; min-width: 0; }
#messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; white-space: pre-wrap; line-height: 1.45; }
.msg.you { align-self: flex-end; background: var(--you); }
.msg.dae { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); }
.msg.err { align-self: flex-start; background: #3d1418; border: 1px solid var(--err); color: #ffb4ac; }
#side { flex: 1; border-left: 1px solid var(--border); background: var(--panel);
  display: flex; flex-direction: column; min-width: 0; }
#side h2 { font-size: 12px; text-transform: uppercase; color: var(--muted); margin: 0; padding: 10px 14px;
  border-bottom: 1px solid var(--border); letter-spacing: 1px; }
#activity { flex: 1; overflow-y: auto; padding: 10px 14px; font-size: 13px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
.act { margin-bottom: 6px; word-break: break-word; }
.act .t { color: var(--accent); }
.act .res { color: var(--dae); }
.act .dec { color: var(--warn); }
form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); background: var(--panel); }
#input { flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; font-size: 14px; resize: none; }
button { background: var(--you); color: #fff; border: 0; border-radius: 8px; padding: 0 18px;
  font-size: 14px; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.approval { align-self: flex-start; max-width: 80%; background: #3a2d0a; border: 1px solid var(--warn);
  border-radius: 12px; padding: 12px 14px; }
.approval .row { margin-top: 8px; display: flex; gap: 8px; }
.approval code { color: var(--muted); font-size: 12px; }
.approval .allow { background: var(--dae); }
.approval .deny { background: var(--err); }
.foot { padding: 4px 16px 10px; color: var(--muted); font-size: 12px; }
"""

_CHAT_HTML = (
    """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Daedalus</title><style>"""
    + _STYLE
    + """</style></head><body>
<header>
  <h1>Daedalus</h1>
  <span class=meta id=meta>connecting…</span>
  <a href="/trace" target="_blank">Trace Viewer ↗</a>
</header>
<main>
  <section id=chat>
    <div id=messages></div>
    <form id=form>
      <textarea id=input rows=1 placeholder="Ask Daedalus…  (Enter to send, Shift+Enter for newline)"></textarea>
      <button id=send type=submit>Send</button>
    </form>
    <div class=foot id=tokens></div>
  </section>
  <aside id=side>
    <h2>Activity</h2>
    <div id=activity></div>
  </aside>
</main>
<script>
const messages = document.getElementById('messages');
const activity = document.getElementById('activity');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
const tokens = document.getElementById('tokens');
const meta = document.getElementById('meta');
let runTokens = 0, busy = false;

fetch('/api/info').then(r => r.json()).then(d => {
  meta.textContent = `provider ${d.provider} · model ${d.model}`;
}).catch(() => { meta.textContent = ''; });

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
let ws;
function connect() {
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setTimeout(connect, 1000); };
}
connect();

function bubble(cls, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}
function clip(v, n=160) {
  let s = typeof v === 'string' ? v : JSON.stringify(v);
  s = (s || '').replace(/\\s+/g, ' ');
  return s.length > n ? s.slice(0, n) + '…' : s;
}
function act(html) {
  const div = document.createElement('div');
  div.className = 'act';
  div.innerHTML = html;
  activity.appendChild(div);
  activity.scrollTop = activity.scrollHeight;
}
function setBusy(b) { busy = b; send.disabled = b; }

function handle(m) {
  if (m.type === 'event') {
    const ev = m.event, d = ev.data || {};
    if (ev.type === 'thought') act(`<span class=t>·</span> ${clip(d.text)}`);
    else if (ev.type === 'tool_call') act(`<span class=t>→ ${d.name}</span> ${clip(d.args)}`);
    else if (ev.type === 'tool_result') act(`<span class=res>←</span> ${clip(d.result)}`);
    else if (ev.type === 'usage') { runTokens += (d.total_tokens || 0); tokens.textContent = `${runTokens} tokens this run`; }
    else if (ev.type === 'decision') {
      if (d.blocked) act(`<span class=dec>blocked: ${d.blocked} (${d.reason||''})</span>`);
      else if (d.halted) act(`<span class=dec>halted: ${clip(d.halted)}</span>`);
      else if (d.skills_matched) act(`<span class=dec>skills: ${d.skills_matched.join(', ')}</span>`);
      else if (d.skill_authored) act(`<span class=dec>new skill: ${d.skill_authored}</span>`);
      else if (d.remembered) act(`<span class=dec>remembered: ${clip(d.remembered.join('; '))}</span>`);
    }
  } else if (m.type === 'final') {
    bubble('dae', m.text); setBusy(false);
  } else if (m.type === 'error') {
    bubble('err', 'error: ' + m.text); setBusy(false);
  } else if (m.type === 'approval_request') {
    renderApproval(m);
  }
}

function renderApproval(m) {
  const card = document.createElement('div');
  card.className = 'approval';
  card.innerHTML = `<b>⚠ Approval needed</b> — <b>${m.tool}</b><br><code>${clip(JSON.stringify(m.args), 220)}</code>
    <div class=row><button class=allow>Allow</button><button class=deny>Deny</button></div>`;
  messages.appendChild(card); messages.scrollTop = messages.scrollHeight;
  const answer = (allow) => { ws.send(JSON.stringify({type:'approval_response', id:m.id, allow})); card.remove(); };
  card.querySelector('.allow').onclick = () => answer(true);
  card.querySelector('.deny').onclick = () => answer(false);
}

function submit() {
  const text = input.value.trim();
  if (!text || busy || !ws || ws.readyState !== 1) return;
  bubble('you', text);
  ws.send(JSON.stringify({type:'message', text}));
  input.value = ''; runTokens = 0; tokens.textContent = '';
  setBusy(true);
  act(`<span class=dec>— run: ${clip(text, 60)}</span>`);
}
form.addEventListener('submit', (e) => { e.preventDefault(); submit(); });
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
input.focus();
</script>
</body></html>"""
)

_TRACE_HTML = (
    """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Daedalus · Trace Viewer</title><style>"""
    + _STYLE
    + """
.run { border: 1px solid var(--border); border-radius: 10px; margin: 12px 16px; background: var(--panel); }
.run > summary { cursor: pointer; padding: 12px 16px; font-family: ui-monospace, monospace; }
.run > summary .rid { color: var(--accent); }
.run > summary .stat { color: var(--muted); margin-left: 10px; font-size: 13px; }
.ev { padding: 6px 16px; border-top: 1px solid var(--border); font-family: ui-monospace, monospace; font-size: 13px; }
.ev .ty { display: inline-block; min-width: 110px; color: var(--accent); }
.ev .st { color: var(--muted); }
.ev pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word; color: var(--muted); }
#wrap { overflow-y: auto; }
.empty { color: var(--muted); padding: 24px; text-align: center; }
</style></head><body>
<header><h1>Daedalus</h1><span class=meta>Trace Viewer</span><a href="/">← Chat</a></header>
<div id=wrap></div>
<script>
const wrap = document.getElementById('wrap');
function esc(s){ return (s==null?'':String(s)).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
async function load() {
  const data = await (await fetch('/api/trace')).json();
  if (!data.runs.length) { wrap.innerHTML = '<div class=empty>No runs recorded yet. Chat with Daedalus, then refresh.</div>'; return; }
  wrap.innerHTML = '';
  for (const run of data.runs) {
    const det = document.createElement('details'); det.className = 'run';
    const when = run.ts ? new Date(run.ts*1000).toLocaleString() : '';
    let evs = '';
    for (const e of run.events) {
      const d = e.data || {};
      evs += `<div class=ev><span class=ty>${esc(e.type)}</span><span class=st>step ${e.step}</span>
        <pre>${esc(JSON.stringify(d))}</pre></div>`;
    }
    det.innerHTML = `<summary><span class=rid>${esc(run.run_id)}</span>
      <span class=stat>${run.events.length} events · ${run.tokens} tokens · $${(run.cost_usd||0).toFixed(4)} · ${esc(when)}</span></summary>${evs}`;
    wrap.appendChild(det);
  }
}
load();
</script>
</body></html>"""
)
