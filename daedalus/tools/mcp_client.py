"""MCP client — graft tools from Model Context Protocol servers into the registry.

The Model Context Protocol (MCP) is an open standard for exposing tools/resources to
an agent over a small JSON-RPC wire. A growing ecosystem of servers (filesystem,
GitHub, Slack, Postgres, Puppeteer, …) speaks it. This module lets Daedalus consume
any of them with **zero core changes**: list the servers in ``MCP_SERVERS`` and their
tools appear in the agent's toolbox, namespaced ``<server>__<tool>`` so two servers
can't collide.

Why the threading dance: an MCP session is a long-lived async connection (a spawned
subprocess for stdio servers, an HTTP stream for SSE ones) that must stay open for the
life of the agent and is *pinned to the event loop that created it*. But ``_build_agent``
is synchronous and is called from several places (the sync REPL bootstrap, async
surfaces, tests). So we run all MCP I/O on a **dedicated background event-loop thread**:

  * :meth:`MCPManager.start` (sync) spins up that thread, opens every session inside a
    single long-lived "serve" coroutine, and returns the discovered tools.
  * Each tool's ``func`` bridges a call from *the agent's* loop to the MCP loop via
    :func:`asyncio.run_coroutine_threadsafe` + :func:`asyncio.wrap_future` — so the
    agent never blocks a thread waiting on a remote tool.
  * :meth:`MCPManager.stop` signals the serve coroutine to unwind, which closes every
    session **on the same task that opened it** (anyio requires this), then stops the
    loop and joins the thread.

The ``mcp`` SDK is an optional dependency (the ``[mcp]`` extra). Everything here imports
it lazily, and :meth:`MCPManager.start` degrades to a friendly no-op when it's absent or
when no servers are configured — so the core install and the offline test suite stay
green without it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from typing import Any

from .registry import Tool

# How long to wait for all configured servers to connect before giving up (seconds).
# A stdio server has to spawn a subprocess (sometimes `npx`-installing it first), so
# this is generous; a hung server shouldn't wedge agent startup forever.
_CONNECT_TIMEOUT = 60.0


def _warn(message: str) -> None:
    """Surface an MCP setup problem without crashing agent startup."""
    print(f"[mcp] {message}")


def _extract_text(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into the plain string the loop feeds back.

    A tool result is a list of typed content blocks (text, image, embedded resource).
    We join the text, describe non-text blocks rather than dropping them silently, and
    prefix an error marker if the server flagged the call as failed — kept as a pure
    function so it's trivially testable without the SDK or a live server.
    """
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            kind = getattr(block, "type", None) or block.__class__.__name__
            parts.append(f"[{kind} content]")
    body = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"ERROR from MCP tool: {body or 'unknown error'}"
    return body or "(no output)"


class MCPManager:
    """Owns the background loop and the live MCP sessions for one agent.

    Construct it with the ``MCP_SERVERS`` config list, call :meth:`start` to connect and
    receive the tools (register them on the :class:`ToolRegistry`), and :meth:`stop` at
    teardown. Safe to construct and start with an empty list — it does nothing.
    """

    def __init__(self, servers: list[dict[str, Any]] | None):
        self._servers = list(servers or [])
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serve_cf: concurrent.futures.Future | None = None
        self._shutdown: asyncio.Event | None = None
        self._sessions: dict[str, Any] = {}
        self.tool_names: list[str] = []

    # ---- lifecycle (synchronous, called from _build_agent) -------------------

    def start(self) -> list[Tool]:
        """Connect to every configured server and return their tools (namespaced).

        A no-op returning ``[]`` when no servers are configured or the ``mcp`` SDK isn't
        installed. A single server failing to connect is logged and skipped — the others
        (and the rest of the agent) still come up.
        """
        if not self._servers:
            return []
        try:
            import mcp  # noqa: F401  (probe: is the optional SDK installed?)
        except ImportError:
            _warn(
                f"{len(self._servers)} MCP server(s) configured but the 'mcp' package "
                'isn\'t installed. Install the extra:  uv pip install -e ".[mcp]"'
            )
            return []

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="dae-mcp", daemon=True)
        self._thread.start()

        # A plain concurrent Future bridges the connect result back to this (sync) thread;
        # the serve coroutine sets it once every session is open (or on first failure).
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._serve_cf = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        try:
            tools = ready.result(timeout=_CONNECT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - connecting is best-effort
            _warn(f"failed to connect to MCP servers: {exc}")
            self.stop()
            return []
        self.tool_names = [t.name for t in tools]
        if tools:
            _warn(f"connected — {len(tools)} tool(s): {', '.join(self.tool_names)}")
        return tools

    def stop(self) -> None:
        """Unwind the sessions and tear down the background loop/thread (idempotent)."""
        loop = self._loop
        if loop is None:
            return
        # Ask the serve coroutine to exit; it closes every session on its own task
        # (anyio forbids closing a cancel scope from a different task).
        if self._shutdown is not None:
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(self._shutdown.set)
        if self._serve_cf is not None:
            with contextlib.suppress(Exception):
                self._serve_cf.result(timeout=10)
        with contextlib.suppress(Exception):
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        with contextlib.suppress(Exception):
            loop.close()
        self._loop = None
        self._thread = None
        self._serve_cf = None
        self._shutdown = None
        self._sessions.clear()

    # ---- the background event loop -------------------------------------------

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        """Open all sessions, hand back the tools, then idle until asked to shut down.

        The whole session lifetime lives in this one coroutine/task: an ``AsyncExitStack``
        holds every open transport + session, and unwinds here when ``_shutdown`` is set —
        so contexts are always exited on the task that entered them.
        """
        from contextlib import AsyncExitStack

        try:
            async with AsyncExitStack() as stack:
                tools: list[Tool] = []
                for server in self._servers:
                    name = str(server.get("name") or "mcp")
                    try:
                        tools.extend(await self._connect_one(stack, name, server))
                    except Exception as exc:  # noqa: BLE001 - skip a bad server, keep the rest
                        _warn(f"server '{name}' failed to connect: {exc}")
                self._shutdown = asyncio.Event()
                ready.set_result(tools)
                await self._shutdown.wait()
        except Exception as exc:  # noqa: BLE001 - report a connect-time failure to start()
            if not ready.done():
                ready.set_exception(exc)

    async def _connect_one(self, stack: Any, name: str, server: dict[str, Any]) -> list[Tool]:
        """Open one server's transport + session, initialize it, and list its tools."""
        from mcp import ClientSession

        if server.get("url"):
            from mcp.client.sse import sse_client

            transport = await stack.enter_async_context(sse_client(server["url"]))
        else:
            command = server.get("command")
            if not command:
                raise ValueError("each MCP server needs a 'url' (SSE) or a 'command' (stdio)")
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=str(command),
                args=[str(a) for a in server.get("args", [])],
                env=server.get("env") or None,
            )
            transport = await stack.enter_async_context(stdio_client(params))

        read, write = transport[0], transport[1]
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session

        listed = await session.list_tools()
        requires_approval = bool(server.get("requires_approval", False))
        return [self._make_tool(name, t, requires_approval) for t in listed.tools]

    # ---- turning a remote tool into a local :class:`Tool` --------------------

    def _make_tool(self, server_name: str, mcp_tool: Any, requires_approval: bool) -> Tool:
        """Wrap one server-advertised tool as a registry :class:`Tool`.

        The name is namespaced ``<server>__<tool>`` so servers can't shadow each other or
        a built-in. The wrapper ``func`` is async and simply bridges the call to the MCP
        loop; the JSON schema the server advertises is forwarded to the model verbatim.
        """
        full_name = f"{server_name}__{mcp_tool.name}"
        schema = getattr(mcp_tool, "inputSchema", None) or {"type": "object", "properties": {}}
        description = getattr(mcp_tool, "description", None) or f"MCP tool {full_name}"

        async def _func(**arguments: Any) -> str:
            return await self._call(server_name, mcp_tool.name, arguments)

        return Tool(
            name=full_name,
            description=description,
            parameters=schema,
            func=_func,
            requires_approval=requires_approval,
        )

    async def _call(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Bridge a tool call from the agent's loop onto the MCP loop and await it.

        ``run_coroutine_threadsafe`` schedules the work on the MCP loop; ``wrap_future``
        turns the returned cross-thread future into one the *agent's* loop can await — so
        we hand control back to the agent's loop instead of blocking its thread.
        """
        loop = self._loop
        if loop is None:
            return f"ERROR: MCP manager for '{server_name}' is not running"
        cf = asyncio.run_coroutine_threadsafe(self._invoke(server_name, tool_name, arguments), loop)
        try:
            return await asyncio.wrap_future(cf)
        except Exception as exc:  # noqa: BLE001 - surface to the model, never crash the loop
            return f"ERROR calling MCP tool {server_name}__{tool_name}: {exc}"

    async def _invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run the actual ``call_tool`` on the MCP loop (where the session lives)."""
        session = self._sessions.get(server_name)
        if session is None:
            return f"ERROR: MCP server '{server_name}' is not connected"
        result = await session.call_tool(tool_name, arguments)
        return _extract_text(result)


def build_mcp_manager(settings: Any) -> MCPManager:
    """Construct a manager from ``settings.mcp_servers`` (the entry point _build_agent uses)."""
    return MCPManager(getattr(settings, "mcp_servers", None))
