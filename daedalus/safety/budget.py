"""The spend governor and the kill switch.

Two safety limits the agent loop consults *before every model call*:

  * **Daily budget** — estimated spend accumulates into a small JSON ledger keyed by
    date (so the cap survives restarts). When today's spend reaches
    ``DAILY_BUDGET_USD`` the loop halts with a clear message. With the default
    ``COST_PER_1K_TOKENS=0`` (free/local models) spend is always $0 and the cap never
    trips — it only bites for paid providers, exactly when you want it to.

  * **Kill switch** — a sentinel file at ``~/.dae/STOP``. While it exists, every run
    halts immediately. ``/stop`` in any surface creates it; ``/resume`` removes it.
    Because it's a file on disk, it stops *all* runs — interactive and scheduled —
    even ones in other processes.

Both surface as exceptions the loop catches and turns into a graceful stop, never a
crash. Accounting is best-effort (a token estimate, not a billing system); the point
is a dependable ceiling, not accountant-grade precision.
"""

from __future__ import annotations

import json
import threading
from datetime import date

from ..config import Settings
from ..core.llm import Usage


class BudgetError(RuntimeError):
    """Base for budget/kill-switch halts — caught by the loop and shown to the user."""


class BudgetExceeded(BudgetError):
    """Raised when today's estimated spend has reached the daily cap."""


class KillSwitchEngaged(BudgetError):
    """Raised when the stop-file is present (a manual emergency stop)."""


class BudgetGovernor:
    """Tracks spend and enforces the daily cap + kill switch."""

    def __init__(
        self,
        settings: Settings,
        *,
        daily_cap_usd: float | None = None,
        cost_per_1k_tokens: float | None = None,
    ):
        home = settings.ensure_home()
        self.daily_cap_usd = settings.daily_budget_usd if daily_cap_usd is None else daily_cap_usd
        self.rate = (
            settings.cost_per_1k_tokens if cost_per_1k_tokens is None else cost_per_1k_tokens
        )
        self.stop_file = home / "STOP"
        self.ledger_path = home / "budget.json"
        self._lock = threading.Lock()
        self.session_tokens = 0
        self.session_cost_usd = 0.0

    # ---- accounting ----------------------------------------------------------

    def _estimate(self, usage: Usage) -> float:
        """Dollar cost of one model call: the provider's figure if any, else tokens×rate."""
        if usage.cost_usd:
            return usage.cost_usd
        return (usage.total_tokens / 1000.0) * self.rate

    def record(self, usage: Usage) -> None:
        """Add one call's usage to the session totals and today's persisted ledger."""
        cost = self._estimate(usage)
        with self._lock:
            self.session_tokens += usage.total_tokens
            self.session_cost_usd += cost
            if cost:
                ledger = self._load_ledger()
                ledger[date.today().isoformat()] = round(
                    ledger.get(date.today().isoformat(), 0.0) + cost, 6
                )
                self._save_ledger(ledger)

    def _load_ledger(self) -> dict[str, float]:
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_ledger(self, ledger: dict[str, float]) -> None:
        try:
            self.ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        except OSError:
            pass  # accounting is best-effort; never crash a run over a write hiccup

    def daily_cost_usd(self) -> float:
        return self._load_ledger().get(date.today().isoformat(), 0.0)

    def remaining_usd(self) -> float:
        return max(0.0, self.daily_cap_usd - self.daily_cost_usd())

    # ---- kill switch ---------------------------------------------------------

    def is_killed(self) -> bool:
        return self.stop_file.exists()

    def engage_kill_switch(self) -> None:
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text("stopped\n", encoding="utf-8")

    def release_kill_switch(self) -> bool:
        """Remove the stop-file. Returns True if one was actually present."""
        existed = self.stop_file.exists()
        self.stop_file.unlink(missing_ok=True)
        return existed

    # ---- the gate the loop calls --------------------------------------------

    def check(self) -> None:
        """Raise if the run must not continue. Called before each model call."""
        if self.is_killed():
            raise KillSwitchEngaged("kill switch engaged (run `/resume` to clear it)")
        if self.daily_cap_usd > 0 and self.daily_cost_usd() >= self.daily_cap_usd:
            raise BudgetExceeded(
                f"daily budget of ${self.daily_cap_usd:.2f} reached "
                f"(spent ${self.daily_cost_usd():.4f})"
            )

    def snapshot(self) -> dict[str, float | bool]:
        """A small dict for the ``/budget`` command to render."""
        return {
            "daily_cap_usd": self.daily_cap_usd,
            "daily_cost_usd": round(self.daily_cost_usd(), 6),
            "remaining_usd": round(self.remaining_usd(), 6),
            "session_tokens": self.session_tokens,
            "session_cost_usd": round(self.session_cost_usd, 6),
            "killed": self.is_killed(),
        }
