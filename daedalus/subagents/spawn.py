"""Subagents — delegate a self-contained subtask to a fresh, bounded child agent.

Some tasks decompose: *"research these three libraries and compare them"* is cleaner as
three focused investigations than one tangled context. A **subagent** is a child
:class:`~daedalus.core.loop.Agent` the parent can spawn for one such subtask. The child
shares the parent's provider, tools, budget, and guardrails — so spend stays under the
*same* daily cap and risky tools still face the *same* approval gate — but starts with an
empty conversation and no memory/skill side-effects, so it can't pollute the parent's
state.

**This is not an autonomous swarm.** It only runs as a tool the parent calls while doing
the work the user explicitly asked for, and it's hard-bounded on two axes:

  * **depth** — how many levels deep spawning may nest (``SUBAGENT_MAX_DEPTH``, default 1:
    the top agent may spawn, its children may not).
  * **count** — how many subagents one *request* may spawn (``SUBAGENT_MAX_CHILDREN``),
    reset at the start of each :meth:`Agent.run` via :meth:`begin_run`.

Both caps fail safe: hitting one returns a plain message to the model (so it adapts), not
an exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.llm import LLMProvider
from ..tools.registry import Tool, ToolRegistry

if TYPE_CHECKING:
    from ..config import Settings
    from ..safety.budget import BudgetGovernor
    from ..safety.guardrails import Guardrails

_SPAWN_TOOL = "spawn_subagent"


class SubagentSpawner:
    """Builds the ``spawn_subagent`` tool and runs depth/count-capped child agents.

    Construct one per session in ``_build_agent``, register :meth:`tool` on the registry,
    and pass the spawner to the :class:`Agent` so it can call :meth:`begin_run` each turn.
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        base_registry: ToolRegistry,
        *,
        budget: BudgetGovernor | None = None,
        guardrails: Guardrails | None = None,
        max_depth: int = 1,
        max_children: int = 3,
    ):
        self.provider = provider
        self.settings = settings
        self.base_registry = base_registry
        self.budget = budget
        self.guardrails = guardrails
        self.max_depth = max(0, max_depth)
        self.max_children = max(0, max_children)
        self._spawned = 0

    def begin_run(self) -> None:
        """Reset the per-request child counter (called at the top of every ``Agent.run``)."""
        self._spawned = 0

    def tool(self, depth: int = 0) -> Tool:
        """Return the ``spawn_subagent`` tool bound to ``depth`` (0 for the top agent)."""

        async def spawn_subagent(task: str) -> str:
            return await self._spawn(task, depth)

        return Tool(
            name=_SPAWN_TOOL,
            description=(
                "Delegate a focused, self-contained subtask to a fresh subagent. The "
                "subagent has the same tools but no memory of this conversation, so give "
                "it everything it needs in 'task'. It returns its final answer as text. "
                "Use it to isolate or parallelize a chunk of work; do NOT use it for the "
                "whole request."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "the complete, standalone instruction for the subagent",
                    }
                },
                "required": ["task"],
            },
            func=spawn_subagent,
            requires_approval=False,
        )

    # ---- internals -----------------------------------------------------------

    async def _spawn(self, task: str, depth: int) -> str:
        if depth >= self.max_depth:
            return "(subagent refused: maximum delegation depth reached)"
        if self._spawned >= self.max_children:
            return (
                f"(subagent refused: this request already spawned the maximum of "
                f"{self.max_children} subagents)"
            )
        self._spawned += 1

        # Import here to avoid an import cycle (loop -> ... -> subagents -> loop).
        from ..core.loop import Agent

        child = Agent(
            self.provider,
            self.settings,
            tools=self._child_registry(depth),
            budget=self.budget,  # shared: subagent spend counts against the same cap
            guardrails=self.guardrails,  # shared: risky tools still need approval
            # Deliberately no memory/skills/subagents handle: a child can't write to the
            # parent's long-term memory, author skills, or (beyond max_depth) spawn further.
        )
        try:
            return await child.run(task)
        except Exception as exc:  # noqa: BLE001 - a child failure is an observation, not a crash
            return f"(subagent error: {exc})"

    def _child_registry(self, depth: int) -> ToolRegistry:
        """A child gets the parent's tools, minus spawn — plus a deeper spawn tool only
        while we're still under the depth cap."""
        tools = [
            self.base_registry.get(name)
            for name in self.base_registry.names()
            if name != _SPAWN_TOOL
        ]
        child_tools: list[Tool] = [t for t in tools if t is not None]
        if depth + 1 < self.max_depth:
            child_tools.append(self.tool(depth + 1))
        return ToolRegistry(child_tools)
