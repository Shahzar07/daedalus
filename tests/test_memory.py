"""Memory: store round-trips, recall, persistence across restarts, and the
post-turn 'what's worth remembering?' reflection wired into the loop."""

import pytest

from daedalus.config import Settings, get_settings
from daedalus.core.llm import MockProvider, Response
from daedalus.core.loop import Agent
from daedalus.memory import files as memory_files
from daedalus.memory.store import MemoryStore


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point ~/.dae at a temp dir so MEMORY.md/USER.md and the db stay off real disk."""
    monkeypatch.setenv("DAE_HOME", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_remember_then_recall(tmp_path):
    store = MemoryStore(tmp_path / "state.db")
    store.remember("The user's name is Ada and she codes in Python.")
    assert store.recall("what is the user's name") == [
        "The user's name is Ada and she codes in Python."
    ]
    store.close()


def test_recall_survives_a_restart(tmp_path):
    """The headline M3 demo: a fact written in one process is recalled in the next."""
    db = tmp_path / "state.db"
    first = MemoryStore(db)
    first.remember("The project lives in C:/code/daedalus.")
    first.close()

    # A brand-new store object == a fresh process pointed at the same file.
    second = MemoryStore(db)
    assert any("daedalus" in m for m in second.recall("where does the project live"))
    second.close()


def test_recall_returns_nothing_for_unrelated_query(tmp_path):
    store = MemoryStore(tmp_path / "state.db")
    store.remember("The user prefers tabs over spaces.")
    assert store.recall("the weather forecast for tomorrow") == []
    store.close()


def test_recent_lists_newest_first(tmp_path):
    store = MemoryStore(tmp_path / "state.db")
    store.remember("first fact")
    store.remember("second fact")
    recent = store.recent(limit=10)
    assert recent[0] == "second fact"
    assert store.count() == 2
    store.close()


def test_blank_memory_is_ignored(tmp_path):
    store = MemoryStore(tmp_path / "state.db")
    store.remember("   ")
    assert store.count() == 0
    store.close()


def test_markdown_files_seed_and_append(home):
    memory_files.ensure_seeded()
    assert "Daedalus Memory" in memory_files.read_memory_md()
    assert "About the User" in memory_files.read_user_md()
    # SOUL.md (the editable persona) is seeded too and loaded every turn.
    assert "Daedalus" in memory_files.read_soul_md()

    memory_files.append_memory_md("The user works in the Pacific timezone.")
    assert "Pacific timezone" in memory_files.read_memory_md()


def test_soul_md_drives_the_system_prompt(home):
    """An edited SOUL.md becomes the persona; absent one, the built-in identity is used."""
    from daedalus.core.context import build_system_prompt

    # No SOUL.md on disk yet -> fall back to the built-in identity.
    assert "You are Daedalus" in build_system_prompt()

    # A custom SOUL.md is used verbatim as the persona base.
    soul = "# Soul\nI am Talos, a terse maritime navigator. I answer in nautical terms."
    assert "Talos" in build_system_prompt(soul_md=soul)
    assert "You are Daedalus" not in build_system_prompt(soul_md=soul)


async def test_loop_remembers_when_summary_says_so(home):
    """A turn whose reflection emits REMEMBER: lands a durable fact in the store."""
    store = MemoryStore(home / "state.db")
    scripted = [
        Response(text="Nice to meet you, Ada!"),  # the answer
        Response(text="REMEMBER: The user's name is Ada."),  # the reflection
    ]
    agent = Agent(MockProvider(scripted), Settings(model_provider="mock"), memory=store)

    answer = await agent.run("Hi, my name is Ada.")
    assert answer == "Nice to meet you, Ada!"
    assert store.count() == 1
    assert store.recall("user name") == ["The user's name is Ada."]
    store.close()


async def test_loop_keeps_nothing_when_summary_says_none(home):
    store = MemoryStore(home / "state.db")
    scripted = [
        Response(text="2 + 2 is 4."),
        Response(text="NONE"),
    ]
    agent = Agent(MockProvider(scripted), Settings(model_provider="mock"), memory=store)

    await agent.run("What is 2 + 2?")
    assert store.count() == 0
    store.close()


async def test_recalled_fact_is_injected_into_the_prompt(home):
    """Stored facts relevant to the new input should reach the model as context."""
    store = MemoryStore(home / "state.db")
    store.remember("The user's name is Ada.")

    captured: dict[str, object] = {}

    class CapturingProvider(MockProvider):
        async def chat(self, messages, tools=None):
            captured.setdefault("system", messages[0]["content"])
            return await super().chat(messages, tools)

    scripted = [Response(text="Hello again, Ada!"), Response(text="NONE")]
    agent = Agent(CapturingProvider(scripted), Settings(model_provider="mock"), memory=store)

    await agent.run("Do you remember my name?")
    assert "Ada" in captured["system"]
    store.close()
