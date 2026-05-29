"""The web surface: HTTP routes, the trace API, and a live WebSocket round-trip.

Auto-skipped when the optional ``[web]`` extra (FastAPI) isn't installed, so the core
$0 suite stays green everywhere; runs in full for anyone who installed the extra.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from daedalus.config import Settings  # noqa: E402
from daedalus.core.llm import MockProvider, Response, ToolCall  # noqa: E402
from daedalus.core.loop import Agent  # noqa: E402
from daedalus.surfaces.web import _read_runs, create_app  # noqa: E402


def _client(scripted=None, home=None) -> TestClient:
    # Point DAE_HOME at an isolated dir when given, so trace/audit reads don't depend on
    # (or pollute) the developer's real ~/.dae — keeps the suite hermetic on any machine.
    settings = (
        Settings(model_provider="mock", dae_home=home) if home else Settings(model_provider="mock")
    )
    agent = Agent(MockProvider(scripted), settings)
    return TestClient(create_app(agent, settings))


def test_index_and_trace_pages_render():
    client = _client()
    r = client.get("/")
    assert r.status_code == 200 and "Daedalus" in r.text
    assert client.get("/trace").status_code == 200


def test_info_route_reports_provider():
    data = _client().get("/api/info").json()
    assert data["provider"] == "mock"


def test_trace_api_empty_when_no_log(tmp_path):
    # Isolated, empty home => no audit.log => the trace API reports zero runs.
    assert _client(home=tmp_path).get("/api/trace").json() == {"runs": []}


def test_read_runs_groups_and_totals(tmp_path):
    log = tmp_path / "audit.log"
    rows = [
        {
            "ts": 1.0,
            "type": "usage",
            "run_id": "r1",
            "step": 1,
            "data": {"total_tokens": 10, "cost_usd": 0.01},
        },
        {"ts": 1.1, "type": "final", "run_id": "r1", "step": 1, "data": {"text": "hi"}},
        {
            "ts": 2.0,
            "type": "usage",
            "run_id": "r2",
            "step": 1,
            "data": {"total_tokens": 5, "cost_usd": 0.0},
        },
    ]
    log.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    out = _read_runs(log)
    assert [r["run_id"] for r in out["runs"]] == ["r2", "r1"]  # newest first
    r1 = next(r for r in out["runs"] if r["run_id"] == "r1")
    assert r1["tokens"] == 10
    assert len(r1["events"]) == 2
    assert r1["cost_usd"] == pytest.approx(0.01)


def test_websocket_round_trip():
    client = _client()  # no script => MockProvider echoes the input
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "message", "text": "hello"})
        final = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "final":
                final = msg
                break
        assert final is not None and final["text"] == "(mock) hello"


def test_websocket_streams_tool_events():
    scripted = [
        Response(tool_calls=[ToolCall(name="echo", arguments={"text": "x"}, id="c0")]),
        Response(text="done"),
    ]
    client = _client(scripted)
    seen: list[str] = []
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "message", "text": "use a tool"})
        for _ in range(30):
            msg = ws.receive_json()
            if msg["type"] == "event":
                seen.append(msg["event"]["type"])
            elif msg["type"] == "final":
                assert msg["text"] == "done"
                break
    assert "tool_call" in seen
