"""Scheduler: natural-language schedule parsing, job persistence, and a live fire.

The parser and the SQLite store have no heavy dependencies and are tested directly.
The live :class:`Scheduler` (APScheduler) is exercised only if the optional
``[scheduler]`` extra is installed — the relevant test is skipped otherwise so the
core $0 suite stays green everywhere.
"""

from __future__ import annotations

import asyncio

import pytest

from daedalus.config import Settings
from daedalus.core.llm import MockProvider
from daedalus.core.loop import Agent
from daedalus.scheduler.jobs import JobStore, ScheduleError, parse_schedule

# ---- natural-language parsing -----------------------------------------------


@pytest.mark.parametrize(
    "text,kind,args",
    [
        ("every 30 seconds", "interval", {"seconds": 30}),
        ("every 5 minutes", "interval", {"minutes": 5}),
        ("every 2 hours", "interval", {"hours": 2}),
        ("every minute", "interval", {"minutes": 1}),
        ("hourly", "interval", {"hours": 1}),
        ("every hour", "interval", {"hours": 1}),
        ("every week", "interval", {"weeks": 1}),
        ("daily", "cron", {"hour": 0, "minute": 0}),
        ("every day at 9am", "cron", {"hour": 9, "minute": 0}),
        ("daily at 21:30", "cron", {"hour": 21, "minute": 30}),
        ("at 08:00", "cron", {"hour": 8, "minute": 0}),
        ("every monday at 9am", "cron", {"day_of_week": "mon", "hour": 9, "minute": 0}),
        ("every weekday at 7:30", "cron", {"day_of_week": "mon-fri", "hour": 7, "minute": 30}),
        ("cron: 0 9 * * 1-5", "cron", {"crontab": "0 9 * * 1-5"}),
        ("0 9 * * 1-5", "cron", {"crontab": "0 9 * * 1-5"}),
    ],
)
def test_parse_schedule_understands_common_forms(text, kind, args):
    spec = parse_schedule(text)
    assert spec.kind == kind
    assert spec.args == args
    assert spec.human  # always echoes something human-readable


def test_parse_schedule_singularizes_units():
    # "12pm" noon, plural/singular units, and 'secs' all resolve cleanly.
    assert parse_schedule("every 1 second").args == {"seconds": 1}
    assert parse_schedule("daily at 12pm").args == {"hour": 12, "minute": 0}
    assert parse_schedule("daily at 12am").args == {"hour": 0, "minute": 0}


@pytest.mark.parametrize("bad", ["", "whenever", "every 0 minutes", "at 99:99", "every blorp"])
def test_parse_schedule_rejects_garbage(bad):
    with pytest.raises(ScheduleError):
        parse_schedule(bad)


# ---- persistence (survives "restart") ---------------------------------------


def test_jobstore_crud_and_persistence(tmp_path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    job = store.add("summarize my inbox", "every day at 9am")
    assert job.enabled and job.human == "every day at 09:00"

    assert len(store.all()) == 1
    assert store.set_enabled(job.id, False) is True
    assert store.enabled() == []  # disabled jobs are excluded
    store.close()

    # New process / new store object over the same file == survives a restart.
    reopened = JobStore(path)
    again = reopened.get(job.id)
    assert again is not None and again.prompt == "summarize my inbox"
    assert again.enabled is False
    assert reopened.delete(job.id) is True
    assert reopened.get(job.id) is None
    reopened.close()


def test_jobstore_rejects_bad_schedule_on_add(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    with pytest.raises(ScheduleError):
        store.add("do a thing", "whenever I feel like it")
    assert store.all() == []  # nothing persisted on a bad schedule
    store.close()


# ---- live scheduler (needs the [scheduler] extra) ---------------------------


def test_scheduler_fires_job_through_agent(tmp_path):
    pytest.importorskip("apscheduler")
    from daedalus.scheduler.heartbeat import Scheduler

    async def scenario() -> list[str]:
        settings = Settings(model_provider="mock", dae_home=tmp_path)
        agent = Agent(MockProvider(), settings)
        store = JobStore(tmp_path / "jobs.db")
        delivered: list[str] = []

        async def deliver(job, result):
            delivered.append(result)

        sched = Scheduler(agent, store, deliver=deliver)
        await sched.start()
        # Fire every second so the test is quick; wait for one tick.
        sched.add_job("ping", "every 1 second")
        for _ in range(40):
            if delivered:
                break
            await asyncio.sleep(0.1)
        await sched.shutdown()
        store.close()
        return delivered

    delivered = asyncio.run(scenario())
    assert delivered and delivered[0] == "(mock) ping"
