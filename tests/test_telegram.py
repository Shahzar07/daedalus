"""Telegram surface: the pure, network-free logic (chunking, allowlist, renderers).

The live bot needs a token and Telegram's servers, so we don't exercise long-polling
here. We do test everything that decides *what* the bot says and *who* it answers — the
parts most likely to regress. Auto-skipped unless the optional ``[telegram]`` extra is
installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("telegram")

from daedalus.config import Settings  # noqa: E402
from daedalus.core.llm import MockProvider  # noqa: E402
from daedalus.core.loop import Agent  # noqa: E402
from daedalus.safety import BudgetGovernor  # noqa: E402
from daedalus.surfaces.telegram_bot import (  # noqa: E402
    TelegramBot,
    _render_budget,
    _render_jobs,
    chunk_message,
)


def _bot(allow: str = "") -> TelegramBot:
    settings = Settings(model_provider="mock", telegram_allowed_chat_ids=allow)
    return TelegramBot(Agent(MockProvider(), settings), settings)


def test_chunk_message_keeps_short_text_whole():
    assert chunk_message("hello") == ["hello"]
    assert chunk_message("") == ["(no output)"]


def test_chunk_message_splits_long_text_within_limit():
    text = "\n".join(f"line {i}" for i in range(2000))
    chunks = chunk_message(text, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks) == text  # nothing lost or duplicated


def test_chunk_message_hard_splits_a_monster_line():
    chunks = chunk_message("x" * 1200, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks) == "x" * 1200


def test_allowlist_enforced_when_set():
    bot = _bot("123,456")
    assert bot.authorized(123) is True
    assert bot.authorized(456) is True
    assert bot.authorized(999) is False


def test_open_mode_answers_everyone_when_allowlist_empty():
    bot = _bot("")
    assert bot.authorized(123) is True
    assert bot.authorized(999) is True


def test_delivery_target_prefers_allowlist_then_last_chat():
    bot = _bot("777,555")
    assert bot._delivery_target() == 555  # smallest allowed id
    open_bot = _bot("")
    assert open_bot._delivery_target() is None
    open_bot._last_chat_id = 42
    assert open_bot._delivery_target() == 42


def test_render_budget_and_jobs_are_friendly(tmp_path):
    settings = Settings(model_provider="mock", dae_home=tmp_path)
    budget = BudgetGovernor(settings)
    out = _render_budget(budget)
    assert "Budget" in out and "kill switch" in out
    assert _render_jobs(None).startswith("Scheduler not running")
