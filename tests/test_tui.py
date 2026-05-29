"""Textual TUI smoke tests, driven headless with Textual's pilot.

These prove the UI wiring without a real terminal: a typed message reaches the agent
and the answer lands in the transcript, slash commands respond, and the event bus
feeds the Activity pane live.
"""

from textual.widgets import Input, RichLog

from daedalus.config import Settings
from daedalus.core.llm import MockProvider, Response
from daedalus.core.loop import Agent
from daedalus.memory.store import MemoryStore
from daedalus.surfaces.tui import DaedalusApp

_MOCK = Settings(model_provider="mock")


def _agent(scripted=None, memory=None):
    return Agent(MockProvider(scripted), _MOCK, memory=memory)


def _text(log: RichLog) -> str:
    """Flatten a RichLog's rendered lines back to plain text for assertions."""
    return "\n".join(strip.text for strip in log.lines)


async def _submit(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_message_reaches_agent_and_answer_is_shown():
    app = DaedalusApp(_agent(scripted=[Response(text="hello from dae")]), _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "hi there")
        transcript = _text(app.query_one("#transcript", RichLog))
        assert "hi there" in transcript
        assert "hello from dae" in transcript


async def test_help_command_lists_commands():
    app = DaedalusApp(_agent(), _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "/help")
        assert "Commands" in _text(app.query_one("#transcript", RichLog))


async def test_skills_command_lists_loaded_skills(tmp_path):
    from daedalus.skills.engine import Skill, SkillLibrary

    lib = SkillLibrary(tmp_path)
    lib.add(Skill(name="web-research", description="research things"))

    agent = Agent(MockProvider(), _MOCK, skills=lib)
    app = DaedalusApp(agent, _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "/skills")
        assert "web-research" in _text(app.query_one("#transcript", RichLog))


async def test_unknown_command_is_reported():
    app = DaedalusApp(_agent(), _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "/nope")
        assert "unknown command" in _text(app.query_one("#transcript", RichLog))


async def test_memory_command_lists_stored_facts(tmp_path):
    store = MemoryStore(tmp_path / "state.db")
    store.remember("The user's favorite language is Python.")
    app = DaedalusApp(_agent(memory=store), _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "/memory")
        assert "favorite language is Python" in _text(app.query_one("#transcript", RichLog))
    store.close()


async def test_tool_activity_streams_to_the_activity_pane():
    """A tool call should surface in the Activity pane as it happens. Uses an isolated
    in-memory tool so the test never touches the network or disk."""
    from daedalus.core.llm import ToolCall
    from daedalus.tools.registry import Tool, ToolRegistry

    ping = Tool(
        name="ping",
        description="returns pong",
        parameters={"type": "object", "properties": {}},
        func=lambda: "pong",
    )
    scripted = [
        Response(tool_calls=[ToolCall(name="ping", arguments={}, id="c0")]),
        Response(text="done"),
    ]
    agent = Agent(MockProvider(scripted), _MOCK, tools=ToolRegistry([ping]))
    app = DaedalusApp(agent, _MOCK)
    async with app.run_test() as pilot:
        await _submit(app, pilot, "please ping")
        activity = _text(app.query_one("#activity", RichLog))
        assert "ping" in activity
        assert "pong" in activity
