"""Approval gates for tools that can change the world.

Some tools are safe to run unattended (a web search, reading a file). Others can
modify your machine — running a shell command, writing or patching files — and those
carry ``requires_approval=True`` in the registry. Guardrails sits between the loop and
the tool and decides whether such a call may proceed, according to a policy:

  * ``auto`` — allow everything (use only for trusted, non-interactive workloads).
  * ``deny`` — block every approval-required tool outright.
  * ``ask``  — defer to an *approver* callback (the surface asks the human). If no
    approver is wired up (a headless run), ``ask`` fails closed: the call is **denied**.

A denied call isn't an error — the loop feeds the model a "blocked" observation so it
can choose another path or explain that it needs permission.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from ..core.llm import ToolCall

Policy = Literal["ask", "auto", "deny"]


@dataclass(slots=True)
class ApprovalRequest:
    """What the user is being asked to approve."""

    tool: str
    args: dict
    reason: str = "this tool can modify your system"


# An approver is given a request and returns True to allow, False to deny.
Approver = Callable[[ApprovalRequest], Awaitable[bool]]


class _HasApprovalFlag(Protocol):
    requires_approval: bool


class _Registry(Protocol):
    def get(self, name: str) -> _HasApprovalFlag | None: ...


class Guardrails:
    """Authorize (or block) tool calls per policy. Wired into :class:`Agent`."""

    def __init__(
        self,
        registry: _Registry | None,
        policy: Policy = "ask",
        approver: Approver | None = None,
    ):
        self.registry = registry
        self.policy: Policy = policy
        self.approver = approver

    def _needs_approval(self, name: str) -> bool:
        tool = self.registry.get(name) if self.registry is not None else None
        return bool(tool is not None and getattr(tool, "requires_approval", False))

    async def authorize(self, call: ToolCall) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for one tool call."""
        if not self._needs_approval(call.name):
            return True, ""
        if self.policy == "auto":
            return True, "auto-approved by policy"
        if self.policy == "deny":
            return False, "blocked by policy (approval required)"
        # policy == "ask"
        if self.approver is None:
            return False, "approval required but no approver is available (denied)"
        req = ApprovalRequest(tool=call.name, args=dict(call.arguments))
        try:
            allowed = await self.approver(req)
        except Exception:  # noqa: BLE001 - a broken approver must fail closed, not crash
            return False, "approval prompt failed (denied)"
        return (True, "approved by user") if allowed else (False, "denied by user")
