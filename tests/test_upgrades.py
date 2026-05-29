"""M10 upgrades: memory graph, semantic index, reflection critic, subagents.

All hermetic and $0 — the only heavy piece (sentence-transformers) is exercised through
its no-op path; its model download is never triggered in the test suite.
"""

from __future__ import annotations

import pytest

from daedalus.config import Settings
from daedalus.core.llm import MockProvider, Response
from daedalus.core.loop import Agent
from daedalus.memory.graph import MemoryGraph, extract_triples
from daedalus.memory.semantic import SemanticIndex, _pack, _unpack
from daedalus.reflection.critic import Critic
from daedalus.subagents.spawn import SubagentSpawner
from daedalus.tools.registry import build_registry

# ---- memory graph -----------------------------------------------------------


@pytest.mark.parametrize(
    "fact,expected",
    [
        ("Alice works at Acme", [("alice", "works_at", "acme")]),
        ("Bob is a teacher", [("bob", "is", "teacher")]),
        ("Carol lives in Berlin", [("carol", "lives_in", "berlin")]),
        ("The user's favourite colour is blue", [("the user", "favourite_colour", "blue")]),
        ("just some rambling text", []),
        ("", []),
    ],
)
def test_extract_triples(fact, expected):
    assert extract_triples(fact) == expected


def test_memory_graph_add_related_and_persist(tmp_path):
    db = tmp_path / "state.db"
    g = MemoryGraph(db)
    assert g.add_fact("Alice works at Acme") == 1
    assert g.add_fact("nothing structured here") == 0
    assert any("acme" in line for line in g.related("Acme"))
    assert g.related("totally unrelated query xyz") == []
    g.close()

    # Reopen the same file == survives a restart.
    again = MemoryGraph(db)
    assert again.count() == 1
    assert again.neighbors("alice") == [("alice", "works_at", "acme")]
    again.close()


# ---- semantic index (no-op path; no model download) -------------------------


def test_semantic_index_is_noop_when_disabled(tmp_path):
    idx = SemanticIndex(tmp_path / "state.db", enabled=False)
    assert idx.available() is False
    assert idx.add("a fact") is False
    assert idx.search("a fact") == []
    assert idx.count() == 0
    idx.close()


def test_semantic_vector_pack_roundtrip():
    vec = [0.1, -0.25, 0.5, 0.0]
    out = _unpack(_pack(vec))
    assert len(out) == len(vec)
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, vec, strict=False))


# ---- reflection critic ------------------------------------------------------


async def test_critic_keeps_answer_when_approved():
    critic = Critic(MockProvider([Response(text="OK")]), max_revisions=1)
    result = await critic.review("q", "the original answer")
    assert result.revised is False
    assert result.answer == "the original answer"


async def test_critic_revises_when_improved():
    critic = Critic(MockProvider([Response(text="a much better answer")]), max_revisions=1)
    result = await critic.review("q", "weak answer")
    assert result.revised is True
    assert result.answer == "a much better answer"


async def test_reflection_revises_final_answer_in_loop():
    # 1st model call: the draft (final). 2nd: the critic's improved answer.
    provider = MockProvider([Response(text="draft answer"), Response(text="polished answer")])
    settings = Settings(model_provider="mock", reflection=True)
    agent = Agent(provider, settings, critic=Critic(provider, max_revisions=1))
    assert await agent.run("question") == "polished answer"
    # The committed transcript reflects the revision, not the draft.
    assert agent.history[-1]["content"] == "polished answer"


async def test_reflection_approval_keeps_draft_in_loop():
    provider = MockProvider([Response(text="good answer"), Response(text="OK")])
    settings = Settings(model_provider="mock", reflection=True)
    agent = Agent(provider, settings, critic=Critic(provider))
    assert await agent.run("question") == "good answer"


# ---- subagents --------------------------------------------------------------


def _spawner(**kw) -> SubagentSpawner:
    return SubagentSpawner(MockProvider(), Settings(model_provider="mock"), build_registry(), **kw)


async def test_subagent_runs_a_child_task():
    spawner = _spawner(max_depth=1, max_children=2)
    spawner.begin_run()
    tool = spawner.tool(depth=0)
    # The child uses the (echoing) mock provider, so it returns the task back.
    assert await tool.func(task="do the thing") == "(mock) do the thing"


async def test_subagent_depth_cap_refuses():
    spawner = _spawner(max_depth=1)
    spawner.begin_run()
    out = await spawner.tool(depth=1).func(task="too deep")
    assert "depth" in out.lower()


async def test_subagent_count_cap_resets_each_run():
    spawner = _spawner(max_depth=1, max_children=1)
    spawner.begin_run()
    tool = spawner.tool(depth=0)
    assert (await tool.func(task="first")).startswith("(mock)")
    assert "maximum" in (await tool.func(task="second")).lower()
    spawner.begin_run()  # a new request resets the per-run budget
    assert (await tool.func(task="third")).startswith("(mock)")


def test_child_registry_drops_spawn_at_depth_cap():
    spawner = _spawner(max_depth=1)
    spawner.base_registry.register([spawner.tool(depth=0)])
    # With max_depth=1, a depth-0 child (depth+1 == 1) is NOT under the cap, so it gets
    # no spawn tool — children can't spawn further.
    child = spawner._child_registry(0)
    assert "spawn_subagent" not in child.names()
    assert "web_search" in child.names()  # but it keeps the ordinary tools


def test_child_registry_keeps_spawn_below_depth_cap():
    spawner = _spawner(max_depth=2)
    spawner.base_registry.register([spawner.tool(depth=0)])
    # With max_depth=2, a depth-0 child (depth+1 == 1 < 2) still gets a spawn tool.
    child = spawner._child_registry(0)
    assert "spawn_subagent" in child.names()
