"""A tiny in-process event bus — the agent's nervous system.

Every interesting moment in a run (a thought, a tool call, a tool result, a final
answer, token usage) is published here as an :class:`Event`. Surfaces and tooling
*subscribe* instead of being wired into the loop directly:

    - the TUI subscribes to stream output live (M4)
    - the web trace viewer subscribes to draw the timeline (M7)
    - the audit log subscribes to write an append-only record (M6)

Because everyone listens to the same bus, the loop stays decoupled from *how* its
activity is shown or stored. Events carry a ``run_id`` so a subscriber can filter to
the conversation it cares about.

This is one of the two sanctioned pieces of global state (the other is config).
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    THOUGHT = "thought"  # model's natural-language reasoning / partial text
    TOOL_CALL = "tool_call"  # model asked to run a tool (name + args)
    TOOL_RESULT = "tool_result"  # a tool returned (output or error)
    DECISION = "decision"  # control-flow note (e.g. "max steps reached")
    USAGE = "usage"  # token / cost accounting for a model call
    FINAL = "final"  # the run's final answer
    ERROR = "error"  # something went wrong


@dataclass(slots=True)
class Event:
    type: EventType
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    ts: float = field(default_factory=time.time)


# A subscriber may be sync or async; both are supported.
Subscriber = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Synchronous-fan-out pub/sub. ``emit`` awaits async subscribers so ordering
    is preserved; sync subscribers are called inline. Subscriber exceptions are
    swallowed deliberately — observability must never crash a run."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a listener. Returns an unsubscribe function."""
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return _unsubscribe

    async def emit(self, event: Event) -> None:
        for callback in list(self._subscribers):
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - a bad listener must not break the agent
                pass


_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Process-wide event bus singleton."""
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS
