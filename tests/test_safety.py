"""Safety rails: budget governor, kill switch, guardrails, audit log, and the way
the loop honors all three. Everything here runs hermetically ($0, no network)."""

from __future__ import annotations

import json

import pytest

from daedalus.config import Settings
from daedalus.core.events import Event, EventBus, EventType
from daedalus.core.llm import MockProvider, Response, ToolCall, Usage
from daedalus.core.loop import Agent
from daedalus.safety import (
    AuditLog,
    BudgetExceeded,
    BudgetGovernor,
    Guardrails,
    KillSwitchEngaged,
)


def _settings(tmp_path):
    """A mock-provider Settings whose ~/.dae lives under the test's tmp dir."""
    return Settings(model_provider="mock", dae_home=tmp_path)


# ---- budget governor ---------------------------------------------------------


def test_budget_records_session_and_ledger(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path), cost_per_1k_tokens=2.0)
    gov.record(Usage(total_tokens=500))  # 500/1000 * 2.0 = 1.0
    assert gov.session_tokens == 500
    assert gov.session_cost_usd == pytest.approx(1.0)
    assert gov.daily_cost_usd() == pytest.approx(1.0)  # persisted to the ledger


def test_budget_prefers_provider_cost_figure(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path), cost_per_1k_tokens=99.0)
    gov.record(Usage(total_tokens=1000, cost_usd=0.25))  # provider figure wins over rate
    assert gov.session_cost_usd == pytest.approx(0.25)


def test_budget_check_trips_at_cap(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path), daily_cap_usd=0.05, cost_per_1k_tokens=1.0)
    gov.record(Usage(total_tokens=60))  # 0.06 > 0.05
    with pytest.raises(BudgetExceeded):
        gov.check()


def test_budget_zero_cap_never_trips(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path), daily_cap_usd=0.0, cost_per_1k_tokens=1.0)
    gov.record(Usage(total_tokens=10_000))
    gov.check()  # must not raise — free/local models run uncapped


def test_kill_switch_roundtrip(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path))
    assert not gov.is_killed()
    gov.engage_kill_switch()
    assert gov.is_killed()
    with pytest.raises(KillSwitchEngaged):
        gov.check()
    assert gov.release_kill_switch() is True
    assert not gov.is_killed()
    assert gov.release_kill_switch() is False  # already gone


def test_budget_snapshot_shape(tmp_path):
    gov = BudgetGovernor(_settings(tmp_path), daily_cap_usd=1.0)
    snap = gov.snapshot()
    assert set(snap) == {
        "daily_cap_usd",
        "daily_cost_usd",
        "remaining_usd",
        "session_tokens",
        "session_cost_usd",
        "killed",
    }


# ---- guardrails --------------------------------------------------------------


class _Tool:
    def __init__(self, requires_approval: bool):
        self.requires_approval = requires_approval


class _Registry:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, name):
        return self._m.get(name)


_REG = _Registry({"danger": _Tool(True), "safe": _Tool(False)})


def _call(name="danger"):
    return ToolCall(name=name, arguments={}, id="c")


async def test_guardrails_safe_tool_always_allowed():
    g = Guardrails(_REG, policy="deny")  # deny still lets a non-approval tool through
    allowed, _ = await g.authorize(_call("safe"))
    assert allowed


async def test_guardrails_auto_allows():
    allowed, reason = await Guardrails(_REG, policy="auto").authorize(_call())
    assert allowed and "auto" in reason


async def test_guardrails_deny_blocks():
    allowed, _ = await Guardrails(_REG, policy="deny").authorize(_call())
    assert not allowed


async def test_guardrails_ask_without_approver_fails_closed():
    allowed, _ = await Guardrails(_REG, policy="ask", approver=None).authorize(_call())
    assert not allowed


async def test_guardrails_ask_approver_yes_and_no():
    async def yes(_req):
        return True

    async def no(_req):
        return False

    assert (await Guardrails(_REG, policy="ask", approver=yes).authorize(_call()))[0] is True
    assert (await Guardrails(_REG, policy="ask", approver=no).authorize(_call()))[0] is False


async def test_guardrails_broken_approver_fails_closed():
    async def boom(_req):
        raise RuntimeError("ui crashed")

    allowed, _ = await Guardrails(_REG, policy="ask", approver=boom).authorize(_call())
    assert not allowed


# ---- audit log ---------------------------------------------------------------


async def test_audit_writes_one_jsonl_line_per_event(tmp_path):
    log = AuditLog(tmp_path / "audit.log")
    bus = EventBus()
    detach = log.attach(bus)
    await bus.emit(Event(EventType.THOUGHT, "run1", {"text": "thinking"}, 1))
    await bus.emit(Event(EventType.FINAL, "run1", {"text": "done"}, 2))
    detach()
    await bus.emit(Event(EventType.THOUGHT, "run1", {"text": "after detach"}, 3))

    lines = (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # the post-detach event is not recorded
    first = json.loads(lines[0])
    assert first["type"] == "thought"
    assert first["run_id"] == "run1"
    assert first["data"]["text"] == "thinking"


async def test_audit_never_raises_on_unserializable_data(tmp_path):
    log = AuditLog(tmp_path / "audit.log")
    circular: dict = {}
    circular["self"] = circular  # json.dumps will raise ValueError on this
    log(Event(EventType.DECISION, "r", {"loop": circular}, 0))  # must not raise
    path = tmp_path / "audit.log"
    assert (path.read_text(encoding="utf-8") if path.exists() else "") == ""


# ---- the loop honoring the rails ---------------------------------------------


class _GatedTools:
    """Implements both the ToolExecutor (schemas/execute) and the guardrails
    registry (get) protocols, exposing one approval-required tool."""

    def __init__(self):
        self.executed: list[ToolCall] = []

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "danger",
                    "description": "a world-changing tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def execute(self, call: ToolCall) -> str:
        self.executed.append(call)
        return "did the dangerous thing"

    def get(self, name):
        return _Tool(True) if name == "danger" else None


async def test_loop_blocks_unapproved_tool_under_deny_policy():
    scripted = [
        Response(tool_calls=[ToolCall(name="danger", arguments={}, id="c0")]),
        Response(text="ok, I won't do that"),
    ]
    tools = _GatedTools()
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    agent = Agent(
        MockProvider(scripted),
        Settings(model_provider="mock"),
        bus=bus,
        tools=tools,
        guardrails=Guardrails(tools, policy="deny"),
    )

    out = await agent.run("do something dangerous")

    assert out == "ok, I won't do that"
    assert tools.executed == []  # the tool was blocked, never ran
    assert any(e.type == EventType.DECISION and "blocked" in e.data for e in seen)


async def test_loop_runs_approved_tool_under_auto_policy():
    scripted = [
        Response(tool_calls=[ToolCall(name="danger", arguments={}, id="c0")]),
        Response(text="all done"),
    ]
    tools = _GatedTools()
    agent = Agent(
        MockProvider(scripted),
        Settings(model_provider="mock"),
        tools=tools,
        guardrails=Guardrails(tools, policy="auto"),
    )
    out = await agent.run("go")
    assert out == "all done"
    assert len(tools.executed) == 1  # auto-approved, so it ran


async def test_loop_halts_when_kill_switch_engaged(tmp_path):
    settings = _settings(tmp_path)
    budget = BudgetGovernor(settings)
    budget.engage_kill_switch()
    # The scripted response must never be reached — check() trips first.
    agent = Agent(MockProvider([Response(text="should not appear")]), settings, budget=budget)
    out = await agent.run("hello")
    assert "should not appear" not in out
    assert "kill switch" in out.lower()


async def test_loop_halts_when_budget_exhausted(tmp_path):
    settings = _settings(tmp_path)
    budget = BudgetGovernor(settings, daily_cap_usd=0.01, cost_per_1k_tokens=1.0)
    budget.record(Usage(total_tokens=100))  # 0.10 spent >> 0.01 cap
    agent = Agent(MockProvider([Response(text="should not appear")]), settings, budget=budget)
    out = await agent.run("hello")
    assert "should not appear" not in out
    assert "budget" in out.lower()


async def test_loop_records_usage_into_budget(tmp_path):
    settings = _settings(tmp_path)
    budget = BudgetGovernor(settings, cost_per_1k_tokens=1.0)
    agent = Agent(
        MockProvider([Response(text="hi", usage=Usage(total_tokens=42))]),
        settings,
        budget=budget,
    )
    await agent.run("hello")
    assert budget.session_tokens == 42
