"""Safety: the guardrails that make Daedalus trustworthy to run on your machine.

Four cooperating pieces, all wired into the one agent loop:

  * :mod:`daedalus.safety.budget`     — a spend governor (per-session + per-day cap)
    and a hard **kill switch** (a stop-file the loop checks before every model call).
  * :mod:`daedalus.safety.guardrails` — approval gates for tools that can change the
    world (shell, file writes); honors an ``ask`` / ``auto`` / ``deny`` policy.
  * :mod:`daedalus.safety.audit`      — an append-only JSONL log of every event, by
    subscribing to the bus. Nothing the agent does goes unrecorded.

Safe-by-default: with no configuration, destructive tools require approval, the loop
can be stopped instantly, and everything is logged locally (never phoned home).
"""

from .audit import AuditLog
from .budget import BudgetError, BudgetExceeded, BudgetGovernor, KillSwitchEngaged
from .guardrails import ApprovalRequest, Guardrails

__all__ = [
    "AuditLog",
    "BudgetError",
    "BudgetExceeded",
    "BudgetGovernor",
    "KillSwitchEngaged",
    "ApprovalRequest",
    "Guardrails",
]
