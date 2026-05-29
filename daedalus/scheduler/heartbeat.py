"""The heartbeat — turns persisted jobs into live, recurring agent runs.

This is the *execution* layer that sits on top of :mod:`daedalus.scheduler.jobs`. On
:meth:`Scheduler.start` it reads every enabled job from the store and registers it with
an APScheduler ``AsyncIOScheduler``; when a trigger fires it hands the job's prompt to
the **same agent the terminal uses**, so a scheduled run inherits memory, skills, and —
crucially — the budget, guardrails, and kill switch. An unattended run has no human to
answer an approval prompt, so ``ask``-policy tools fail closed (the safe default).

Design choices worth seeing:

  * **We keep our own SQLite job table, not APScheduler's job store.** APScheduler's
    persistent stores pickle the job's target function, which would mean pickling the
    live agent — impossible and undesirable. Instead the source of truth is our table;
    on startup we rebuild the in-memory schedule from it. Restart-safe, no pickling.
  * **Scheduled runs don't pollute the chat.** We snapshot and restore ``agent.history``
    around each fire so a 3am job doesn't bleed into your next interactive turn, and we
    serialize through a shared lock so a job and a live chat never interleave.
  * **Delivery is a callback.** Where a result *goes* (a Telegram message, a log line)
    is the surface's business; the heartbeat just runs the agent and calls ``deliver``.

APScheduler lives behind the optional ``[scheduler]`` extra and is imported lazily, so
the core install and the test suite don't need it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..core.events import Event, EventType, get_event_bus
from ..core.loop import Agent
from .jobs import Job, JobStore, ScheduleSpec

if TYPE_CHECKING:  # avoid importing the heavy dep unless the scheduler is actually used
    from apscheduler.triggers.base import BaseTrigger

# A delivery sink: given the job that ran and its result text, push it somewhere.
DeliverFn = Callable[[Job, str], Awaitable[None]]


def build_trigger(spec: ScheduleSpec) -> BaseTrigger:
    """Convert a dependency-free :class:`ScheduleSpec` into an APScheduler trigger.

    Imported lazily so :func:`daedalus.scheduler.jobs.parse_schedule` stays testable
    without the extra installed.
    """
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    if spec.kind == "interval":
        return IntervalTrigger(**spec.args)  # type: ignore[arg-type]
    if "crontab" in spec.args:
        return CronTrigger.from_crontab(str(spec.args["crontab"]))
    return CronTrigger(**spec.args)  # type: ignore[arg-type]


class Scheduler:
    """Runs user-defined jobs on time, against the shared agent.

    Manage jobs through this object (``add_job`` / ``pause`` / ``resume`` / ``delete``)
    and it keeps the persistent store and the live APScheduler in lock-step. Construct
    it, ``await start()`` once a loop is running, and ``await shutdown()`` on the way out.
    """

    def __init__(
        self,
        agent: Agent,
        store: JobStore,
        *,
        deliver: DeliverFn | None = None,
        run_lock: asyncio.Lock | None = None,
    ):
        self.agent = agent
        self.store = store
        self.deliver = deliver
        # Share this lock with the surface (e.g. the Telegram message handler) so a
        # scheduled run and a live chat never run the one agent at the same time.
        self.run_lock = run_lock or asyncio.Lock()
        self._sched = None  # the AsyncIOScheduler, created in start()
        self._started = False

    async def start(self) -> None:
        """Create the APScheduler, register every enabled job, and begin firing."""
        if self._started:
            return
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._sched = AsyncIOScheduler()
        for job in self.store.enabled():
            self._register(job)
        self._sched.start()
        self._started = True

    async def shutdown(self) -> None:
        if self._sched is not None:
            self._sched.shutdown(wait=False)
            self._sched = None
        self._started = False

    # ---- management (keeps store + live scheduler in sync) -------------------

    def add_job(self, prompt: str, schedule: str) -> Job:
        """Persist a new job and, if we're running, schedule it immediately."""
        job = self.store.add(prompt, schedule)  # validates the schedule, raises on garbage
        if self._sched is not None:
            self._register(job)
        return job

    def pause(self, job_id: str) -> bool:
        if not self.store.set_enabled(job_id, False):
            return False
        self._remove_live(job_id)
        return True

    def resume(self, job_id: str) -> bool:
        if not self.store.set_enabled(job_id, True):
            return False
        job = self.store.get(job_id)
        if job is not None and self._sched is not None:
            self._register(job)
        return True

    def delete(self, job_id: str) -> bool:
        self._remove_live(job_id)
        return self.store.delete(job_id)

    def list_jobs(self) -> list[Job]:
        return self.store.all()

    # ---- internals -----------------------------------------------------------

    def _register(self, job: Job) -> None:
        """Add (or replace) one job in the live scheduler from its stored schedule."""
        if self._sched is None:
            return
        try:
            trigger = build_trigger(job.spec())
        except Exception:  # noqa: BLE001 - a malformed stored schedule shouldn't crash startup
            return
        self._sched.add_job(
            self._fire, trigger=trigger, args=[job.id], id=job.id, replace_existing=True
        )

    def _remove_live(self, job_id: str) -> None:
        if self._sched is not None:
            with contextlib.suppress(Exception):
                self._sched.remove_job(job_id)

    async def _fire(self, job_id: str) -> None:
        """A trigger fired: run the job's prompt through the agent, then deliver.

        Wrapped so nothing here can crash the scheduler thread: a failed job records its
        error and moves on. History is isolated and runs are serialized via the lock.
        """
        job = self.store.get(job_id)
        if job is None or not job.enabled:
            return

        run_id = "job-" + uuid.uuid4().hex[:6]
        await get_event_bus().emit(
            Event(EventType.DECISION, run_id, {"scheduled_job": job.id, "prompt": job.prompt}, 0)
        )

        async with self.run_lock:
            saved_history = self.agent.history
            self.agent.history = []  # a scheduled task starts from a clean slate
            try:
                result = await self.agent.run(job.prompt, run_id=run_id)
            except Exception as exc:  # noqa: BLE001 - record + continue; never kill the heartbeat
                result = f"(job failed: {exc})"
            finally:
                self.agent.history = saved_history

        self.store.record_run(job.id, time.time(), result)
        if self.deliver is not None:
            with contextlib.suppress(Exception):
                await self.deliver(job, result)
