"""User-defined scheduled jobs: persistence + natural-language schedule parsing.

This is the *data* layer for the scheduler. Jobs live in their own SQLite file
(``~/.dae/jobs.db``) so they survive restarts — the running scheduler
(:mod:`daedalus.scheduler.heartbeat`) re-registers every enabled job from this table
on startup.

A "job" is just three things: a **prompt** to run, a **schedule** (when), and an
**enabled** flag. When the schedule fires, the heartbeat hands the prompt to the very
same :class:`~daedalus.core.loop.Agent` the terminal uses — so a scheduled run obeys
the budget, guardrails, and kill switch exactly like an interactive one.

Scope line we don't cross: nothing here makes the agent chase goals on its own. A
human created the job and said *when*. This is a cron with a friendlier face, not an
autonomous loop.

Why a separate file from the heartbeat: parsing "every day at 9am" into a trigger has
no dependencies and is a joy to unit-test; the live scheduling (APScheduler) is an
optional extra. Keeping them apart means the parser is always testable and the heavy
dependency stays lazy.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class ScheduleError(ValueError):
    """Raised when a schedule string can't be understood. Carries a friendly message."""


@dataclass(slots=True)
class ScheduleSpec:
    """A normalized, dependency-free description of *when* a job runs.

    ``kind`` is ``"interval"`` or ``"cron"``; ``args`` are the keyword arguments for
    the matching APScheduler trigger (built lazily in :mod:`heartbeat`). ``human`` is a
    plain-English echo we store so ``/jobs`` can show what the user meant.
    """

    kind: Literal["interval", "cron"]
    args: dict[str, object]
    human: str


@dataclass(slots=True)
class Job:
    """One scheduled job as stored on disk."""

    id: str
    prompt: str
    schedule: str  # the original natural-language text (re-parsed on load)
    human: str  # friendly description of the schedule
    enabled: bool
    created_at: float
    last_run: float | None = None
    last_result: str | None = None

    def spec(self) -> ScheduleSpec:
        """Re-parse the stored schedule text into a trigger spec."""
        return parse_schedule(self.schedule)


# --- natural-language schedule parsing ---------------------------------------

_UNIT_SECONDS = {
    "second": "seconds",
    "sec": "seconds",
    "minute": "minutes",
    "min": "minutes",
    "hour": "hours",
    "hr": "hours",
    "day": "days",
    "week": "weeks",
}

_WEEKDAYS = {
    "monday": "mon",
    "mon": "mon",
    "tuesday": "tue",
    "tue": "tue",
    "wednesday": "wed",
    "wed": "wed",
    "thursday": "thu",
    "thu": "thu",
    "friday": "fri",
    "fri": "fri",
    "saturday": "sat",
    "sat": "sat",
    "sunday": "sun",
    "sun": "sun",
}


def _parse_time(text: str) -> tuple[int, int]:
    """Parse a clock time like ``9am``, ``9:30am``, ``21:00``, or ``9`` -> (hour, minute)."""
    text = text.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", text)
    if not m:
        raise ScheduleError(f"could not read the time {text!r} (try '9am' or '21:30')")
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ScheduleError(f"time out of range in {text!r}")
    return hour, minute


def parse_schedule(text: str) -> ScheduleSpec:
    """Turn a natural-language (or raw-cron) schedule into a :class:`ScheduleSpec`.

    Understood forms (case-insensitive)::

        every 30 seconds / every 5 minutes / every 2 hours / every day
        hourly / daily / every hour / every minute
        every day at 9am / daily at 21:30 / at 08:00
        every monday at 9am / every weekday at 7:30 / every sat at 10
        cron: 0 9 * * 1-5         (raw 5-field crontab, with or without the 'cron:' prefix)

    Raises :class:`ScheduleError` (a ``ValueError``) with a friendly message otherwise.
    """
    raw = text.strip()
    if not raw:
        raise ScheduleError("empty schedule")
    s = raw.lower()

    # Raw crontab, e.g. "cron: 0 9 * * 1-5" or just "0 9 * * 1-5".
    crontab = s[5:].strip() if s.startswith("cron:") else s
    if re.fullmatch(r"[\d*/,\-]+(?:\s+[\d*/,\-]+){4}", crontab):
        return ScheduleSpec("cron", {"crontab": crontab}, f"cron({crontab})")

    # Friendly aliases.
    if s in ("hourly", "every hour"):
        return ScheduleSpec("interval", {"hours": 1}, "every hour")
    if s in ("daily", "every day"):
        return ScheduleSpec("cron", {"hour": 0, "minute": 0}, "every day at 00:00")
    if s in ("every minute",):
        return ScheduleSpec("interval", {"minutes": 1}, "every minute")
    if s in ("weekly", "every week"):
        return ScheduleSpec("interval", {"weeks": 1}, "every week")

    # "every N <unit>" / "every <unit>"  -> interval. We capture the whole unit word
    # (greedy ``[a-z]+`` would otherwise eat the plural "s") and singularize it, so
    # "second", "seconds", "sec", and "secs" all resolve to the same trigger.
    m = re.fullmatch(r"every\s+(\d+)?\s*([a-z]+)", s)
    if m:
        word = m.group(2)
        kw = _UNIT_SECONDS.get(word) or _UNIT_SECONDS.get(word.rstrip("s"))
        if kw:
            count = int(m.group(1) or 1)
            if count < 1:
                raise ScheduleError("interval must be at least 1")
            unit_label = kw[:-1] if count == 1 else kw
            return ScheduleSpec("interval", {kw: count}, f"every {count} {unit_label}")

    # "(every day|daily|every <weekday>|every weekday) at <time>"  -> cron
    m = re.fullmatch(r"(?:every\s+)?(day|daily|weekday|[a-z]+)\s+at\s+(.+)", s)
    if m:
        who, when = m.group(1), m.group(2)
        hour, minute = _parse_time(when)
        if who in ("day", "daily"):
            return ScheduleSpec(
                "cron", {"hour": hour, "minute": minute}, f"every day at {hour:02d}:{minute:02d}"
            )
        if who == "weekday":
            return ScheduleSpec(
                "cron",
                {"day_of_week": "mon-fri", "hour": hour, "minute": minute},
                f"every weekday at {hour:02d}:{minute:02d}",
            )
        if who in _WEEKDAYS:
            dow = _WEEKDAYS[who]
            return ScheduleSpec(
                "cron",
                {"day_of_week": dow, "hour": hour, "minute": minute},
                f"every {dow} at {hour:02d}:{minute:02d}",
            )

    # Bare "at <time>" -> daily at that time.
    m = re.fullmatch(r"at\s+(.+)", s)
    if m:
        hour, minute = _parse_time(m.group(1))
        return ScheduleSpec(
            "cron", {"hour": hour, "minute": minute}, f"every day at {hour:02d}:{minute:02d}"
        )

    raise ScheduleError(
        f"could not understand the schedule {raw!r}. Try 'every 30 minutes', "
        "'every day at 9am', 'every monday at 7:30', or a cron like '0 9 * * 1-5'."
    )


# --- persistence -------------------------------------------------------------


class JobStore:
    """SQLite-backed CRUD for scheduled jobs.

    The store is deliberately ignorant of *running* anything — it just remembers what
    the user asked for. The heartbeat owns execution. This split keeps ``dae jobs``
    (manage jobs without a server running) and the live scheduler reading the same
    source of truth.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: APScheduler may fire jobs from a worker thread.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                prompt      TEXT NOT NULL,
                schedule    TEXT NOT NULL,
                human       TEXT NOT NULL,
                enabled     INTEGER NOT NULL DEFAULT 1,
                created_at  REAL NOT NULL,
                last_run    REAL,
                last_result TEXT
            )
            """)
        self._db.commit()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            prompt=row["prompt"],
            schedule=row["schedule"],
            human=row["human"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_run=row["last_run"],
            last_result=row["last_result"],
        )

    def add(self, prompt: str, schedule: str) -> Job:
        """Validate the schedule, then persist a new job. Raises :class:`ScheduleError`."""
        prompt = prompt.strip()
        if not prompt:
            raise ScheduleError("a job needs a prompt to run")
        spec = parse_schedule(schedule)  # validate up front; raises on garbage
        job = Job(
            id=uuid.uuid4().hex[:8],
            prompt=prompt,
            schedule=schedule.strip(),
            human=spec.human,
            enabled=True,
            created_at=time.time(),
        )
        self._db.execute(
            "INSERT INTO jobs (id, prompt, schedule, human, enabled, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (job.id, job.prompt, job.schedule, job.human, job.created_at),
        )
        self._db.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def all(self) -> list[Job]:
        rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        return [self._row_to_job(r) for r in rows]

    def enabled(self) -> list[Job]:
        rows = self._db.execute(
            "SELECT * FROM jobs WHERE enabled = 1 ORDER BY created_at"
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def set_enabled(self, job_id: str, enabled: bool) -> bool:
        cur = self._db.execute(
            "UPDATE jobs SET enabled = ? WHERE id = ?", (1 if enabled else 0, job_id)
        )
        self._db.commit()
        return cur.rowcount > 0

    def record_run(self, job_id: str, when: float, result: str) -> None:
        """Stamp the last-run time and a clipped result (best-effort observability)."""
        self._db.execute(
            "UPDATE jobs SET last_run = ?, last_result = ? WHERE id = ?",
            (when, result[:500], job_id),
        )
        self._db.commit()

    def delete(self, job_id: str) -> bool:
        cur = self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._db.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._db.close()
