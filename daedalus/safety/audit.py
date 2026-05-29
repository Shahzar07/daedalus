"""Append-only audit log.

A tamper-evident-ish record of everything the agent did: each bus event becomes one
JSON line in ``~/.dae/audit.log``. Because it's just a bus subscriber, it captures
thoughts, tool calls, tool results, decisions, usage, and final answers without the
loop knowing it's being watched — the same decoupling the TUI and trace viewer use.

JSONL (one object per line) is append-only and trivially greppable; you can tail it
live while the agent works. Writes are best-effort: a logging failure must never break
a run.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..core.events import Event, EventBus


class AuditLog:
    """A bus subscriber that appends each event to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        """Subscriber entry point — write one line per event (never raises)."""
        record = {
            "ts": event.ts or time.time(),
            "type": str(event.type),
            "run_id": event.run_id,
            "step": event.step,
            "data": event.data,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # auditing is best-effort; never crash a run over a write hiccup

    def attach(self, bus: EventBus) -> Callable[[], None]:
        """Subscribe to ``bus`` and return the unsubscribe function."""
        return bus.subscribe(self)
