"""MCP client: result flattening, tool wrapping, and graceful no-ops.

The pure parts (text extraction, tool namespacing, the empty-config no-op) are tested
directly with no SDK and no network. The thread/loop lifecycle is exercised only when
the optional ``mcp`` SDK is installed — that test connects to a deliberately bogus
server and asserts the manager degrades to ``[]`` and tears down cleanly, never hangs.
"""

from __future__ import annotations

import pytest

from daedalus.config import Settings
from daedalus.tools.mcp_client import MCPManager, _extract_text, build_mcp_manager

# ---- result flattening (_extract_text) --------------------------------------


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ImageBlock:
    # No `.text` attribute on purpose — represents a non-text content block.
    def __init__(self):
        self.type = "image"


class _Result:
    def __init__(self, content, is_error: bool = False):
        self.content = content
        self.isError = is_error


def test_extract_text_joins_text_blocks():
    res = _Result([_TextBlock("line one"), _TextBlock("line two")])
    assert _extract_text(res) == "line one\nline two"


def test_extract_text_describes_non_text_blocks():
    res = _Result([_ImageBlock()])
    assert _extract_text(res) == "[image content]"


def test_extract_text_flags_errors():
    res = _Result([_TextBlock("boom")], is_error=True)
    assert _extract_text(res) == "ERROR from MCP tool: boom"


def test_extract_text_handles_empty_result():
    assert _extract_text(_Result([])) == "(no output)"


# ---- wrapping a server tool as a registry Tool ------------------------------


class _StubTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


def test_make_tool_namespaces_and_forwards_schema():
    mgr = MCPManager([])
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    tool = mgr._make_tool("github", _StubTool("search", "find things", schema), True)
    assert tool.name == "github__search"  # namespaced <server>__<tool>
    assert tool.description == "find things"
    assert tool.parameters == schema
    assert tool.requires_approval is True


def test_make_tool_supplies_defaults_when_server_is_terse():
    mgr = MCPManager([])
    tool = mgr._make_tool("fs", _StubTool("ls", None, None), False)
    assert tool.name == "fs__ls"
    assert tool.description  # a fallback description is always present
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.requires_approval is False


# ---- graceful no-ops (no SDK, no network) -----------------------------------


def test_empty_config_is_a_noop():
    assert MCPManager([]).start() == []
    assert MCPManager(None).start() == []


def test_build_mcp_manager_reads_settings():
    settings = Settings(model_provider="mock")
    mgr = build_mcp_manager(settings)
    assert isinstance(mgr, MCPManager)
    assert mgr.start() == []  # default settings configure no servers


# ---- full lifecycle against a bogus server (needs the [mcp] extra) ----------


def test_bad_server_degrades_and_tears_down_cleanly():
    pytest.importorskip("mcp")
    # A command that cannot be spawned: connecting fails, the server is skipped, and the
    # manager still returns a (empty) list and stops without hanging the test.
    mgr = MCPManager([{"name": "bogus", "command": "dae-nonexistent-binary-xyz"}])
    tools = mgr.start()
    assert isinstance(tools, list) and tools == []
    mgr.stop()  # must not raise or block
