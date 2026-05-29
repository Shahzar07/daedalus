"""Tool registry: discovery, dispatch, safety, and a real two-step loop task."""

import pytest

from daedalus.config import Settings, get_settings
from daedalus.core.llm import MockProvider, Response, ToolCall
from daedalus.core.loop import Agent
from daedalus.tools.registry import build_registry, discover_tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point ~/.dae at a temp dir so file/shell tools can't touch the real disk."""
    monkeypatch.setenv("DAE_HOME", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path / "workspace"
    get_settings.cache_clear()


def test_discovery_finds_builtin_tools():
    names = {t.name for t in discover_tools()}
    assert {"web_search", "files_read", "files_write", "files_patch", "shell_exec"} <= names


def test_schemas_are_openai_shaped():
    schema = build_registry().schemas()[0]
    assert schema["type"] == "function"
    assert "name" in schema["function"] and "parameters" in schema["function"]


async def test_files_write_then_read(workspace):
    reg = build_registry()
    wrote = await reg.execute(
        ToolCall(name="files_write", arguments={"path": "note.txt", "content": "hello dae"})
    )
    assert "wrote" in wrote
    read = await reg.execute(ToolCall(name="files_read", arguments={"path": "note.txt"}))
    assert read == "hello dae"
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hello dae"


async def test_files_patch(workspace):
    reg = build_registry()
    await reg.execute(
        ToolCall(name="files_write", arguments={"path": "p.txt", "content": "foo bar"})
    )
    await reg.execute(
        ToolCall(name="files_patch", arguments={"path": "p.txt", "find": "foo", "replace": "baz"})
    )
    assert (workspace / "p.txt").read_text(encoding="utf-8") == "baz bar"


async def test_path_escape_is_blocked(workspace):
    out = await build_registry().execute(
        ToolCall(name="files_read", arguments={"path": "../../secrets.txt"})
    )
    assert "escapes" in out


async def test_unknown_tool_is_reported():
    out = await build_registry().execute(ToolCall(name="does_not_exist", arguments={}))
    assert "unknown tool" in out


async def test_bad_arguments_are_reported(workspace):
    out = await build_registry().execute(ToolCall(name="files_read", arguments={"wrong": "x"}))
    assert "bad arguments" in out


async def test_two_step_task_actually_writes_file(workspace):
    scripted = [
        Response(
            tool_calls=[
                ToolCall(
                    name="files_write", arguments={"path": "a.txt", "content": "data"}, id="c0"
                )
            ]
        ),
        Response(text="saved it"),
    ]
    agent = Agent(MockProvider(scripted), Settings(model_provider="mock"), tools=build_registry())
    out = await agent.run("save data to a.txt")
    assert out == "saved it"
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "data"


async def test_shell_exec_runs_a_command(workspace):
    out = await build_registry().execute(
        ToolCall(name="shell_exec", arguments={"command": "echo hello-dae"})
    )
    assert "hello-dae" in out
