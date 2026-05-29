"""Tool registry — auto-discovery + schema + dispatch.

The contract is deliberately tiny so adding a capability is frictionless: drop a
module in ``daedalus/tools/`` that defines a ``TOOL`` (or a ``TOOLS`` list) of
:class:`Tool` objects, and the registry finds it at startup. No core edits, no
manual registration list to keep in sync.

A :class:`Tool` is just a Python function plus a JSON-schema description of its
arguments and a ``requires_approval`` flag (enforced by the guardrails in M6). The
:class:`ToolRegistry` exposes exactly the two methods the agent loop's
``ToolExecutor`` protocol needs: :meth:`schemas` and :meth:`execute`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.llm import ToolCall, ToolSchema


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments (OpenAI "parameters")
    func: Callable[..., Any]  # sync or async; called with **arguments
    requires_approval: bool = False

    def to_schema(self) -> ToolSchema:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the discovered tools and runs them on the loop's behalf."""

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def register(self, tools: list[Tool]) -> None:
        """Add tools after construction (used by the MCP client to graft in remote tools).

        Later names win on collision, so a server's tools can't silently shadow a built-in
        without being obvious in the logs the caller emits.
        """
        for tool in tools:
            self._tools[tool.name] = tool

    # ---- ToolExecutor protocol (consumed by core/loop.py) --------------------

    def schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return (
                f"ERROR: unknown tool '{call.name}'. Available: {', '.join(self._tools) or 'none'}"
            )

        try:
            result = tool.func(**call.arguments)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as exc:
            # Almost always the model passed wrong/missing arguments.
            return f"ERROR: bad arguments for {call.name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface tool failures, don't crash
            return f"ERROR in {call.name}: {exc}"

        return result if isinstance(result, str) else str(result)


def discover_tools() -> list[Tool]:
    """Import every public module under ``daedalus.tools`` and collect their tools."""
    import daedalus.tools as pkg

    found: list[Tool] = []
    for module_info in pkgutil.iter_modules(pkg.__path__):
        name = module_info.name
        if name == "registry" or name.startswith("_"):
            continue
        module = importlib.import_module(f"daedalus.tools.{name}")
        if isinstance(getattr(module, "TOOL", None), Tool):
            found.append(module.TOOL)
        for t in getattr(module, "TOOLS", []) or []:
            if isinstance(t, Tool):
                found.append(t)
    return found


def build_registry() -> ToolRegistry:
    """The single entry point the surfaces use to get a ready tool registry."""
    return ToolRegistry(discover_tools())
