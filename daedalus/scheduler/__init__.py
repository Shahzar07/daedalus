"""The scheduler — natural-language cron for **user-defined** recurring tasks.

Two layers, deliberately split:

  * :mod:`daedalus.scheduler.jobs`      — persistence (``~/.dae/jobs.db``) and the
    natural-language schedule parser. No heavy dependency; always importable & testable.
  * :mod:`daedalus.scheduler.heartbeat` — the live runner (APScheduler, behind the
    optional ``[scheduler]`` extra) that fires jobs through the shared agent.

A job is a prompt + a schedule the *user* chose. When it fires, the same agent loop the
terminal uses runs it — so every scheduled run obeys the budget, guardrails, and kill
switch. We never invent goals or run anything the user didn't ask for; this is a cron,
not an autonomous agent.
"""

from .jobs import Job, JobStore, ScheduleError, ScheduleSpec, parse_schedule

__all__ = [
    "Job",
    "JobStore",
    "ScheduleError",
    "ScheduleSpec",
    "parse_schedule",
]
