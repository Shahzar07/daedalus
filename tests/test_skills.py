"""Skills: SKILL.md parsing, matching, persistence, seeding, and the loop wiring
(injection into the prompt + post-task authoring)."""

import pytest

from daedalus.config import Settings, get_settings
from daedalus.core.llm import MockProvider, Response, ToolCall
from daedalus.core.loop import Agent
from daedalus.skills.engine import (
    Skill,
    SkillLibrary,
    build_library,
    parse_skill_md,
    render_skill_md,
)

_SAMPLE = """\
---
name: web-research
description: Research a topic using web search
triggers: [research, look up, investigate]
tags: [research, web]
---
## Steps
1. Search.
2. Synthesize.
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DAE_HOME", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---- parsing ----------------------------------------------------------------


def test_parse_frontmatter_and_body():
    skill = parse_skill_md(_SAMPLE)
    assert skill is not None
    assert skill.name == "web-research"
    assert skill.description == "Research a topic using web search"
    assert skill.triggers == ["research", "look up", "investigate"]
    assert skill.tags == ["research", "web"]
    assert "Synthesize" in skill.body


def test_parse_without_frontmatter_uses_folder_name(tmp_path):
    p = tmp_path / "my-skill" / "SKILL.md"
    p.parent.mkdir(parents=True)
    p.write_text("just a body, no frontmatter", encoding="utf-8")
    skill = parse_skill_md(p.read_text(encoding="utf-8"), p)
    assert skill is not None and skill.name == "my-skill"


def test_render_then_parse_roundtrip():
    original = Skill(
        name="git-commit",
        description="Commit changes",
        body="## Steps\n1. Stage.\n2. Commit.",
        triggers=["commit", "git"],
        tags=["git"],
    )
    reparsed = parse_skill_md(render_skill_md(original))
    assert reparsed is not None
    assert reparsed.name == original.name
    assert reparsed.triggers == original.triggers
    assert "Stage." in reparsed.body


def test_block_list_frontmatter():
    text = "---\nname: x\ndescription: d\ntriggers:\n  - alpha\n  - beta\n---\nbody"
    skill = parse_skill_md(text)
    assert skill is not None and skill.triggers == ["alpha", "beta"]


# ---- library: match / add / remove ------------------------------------------


def test_match_ranks_relevant_skill_first(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill(name="web-research", description="Research a topic using web search",
                  triggers=["research", "investigate"]))  # fmt: skip
    lib.add(Skill(name="git-commit", description="Commit changes to git",
                  triggers=["commit"]))  # fmt: skip

    hits = lib.match("please research the latest news on quantum computing")
    assert hits and hits[0].name == "web-research"


def test_match_returns_empty_when_nothing_relevant(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill(name="git-commit", description="Commit changes to git", triggers=["commit"]))
    assert lib.match("bake a chocolate cake") == []


def test_add_persists_and_reload_finds_it(tmp_path):
    lib = SkillLibrary(tmp_path)
    path = lib.add(Skill(name="my-skill", description="does a thing"))
    assert path.exists()

    fresh = SkillLibrary(tmp_path)  # simulate a restart
    assert fresh.get("my-skill") is not None
    assert fresh.count() == 1


def test_remove_deletes_skill(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill(name="temp", description="temporary"))
    assert lib.remove("temp") is True
    assert lib.get("temp") is None
    assert SkillLibrary(tmp_path).count() == 0  # gone from disk too


# ---- seeding ----------------------------------------------------------------


def test_build_library_seeds_starter_skills(home):
    lib = build_library()
    assert lib.count() >= 5  # the bundled starter library
    assert lib.get("web-research") is not None


# ---- loop integration -------------------------------------------------------


class _FakeTools:
    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def execute(self, call):
        return "search results"


async def test_matched_skill_is_injected_into_prompt(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(
        Skill(
            name="web-research",
            description="Research a topic using web search",
            body="## Steps\n1. Search carefully.",
            triggers=["research", "investigate"],
        )
    )
    captured: dict[str, object] = {}

    class CapturingProvider(MockProvider):
        async def chat(self, messages, tools=None):
            captured.setdefault("system", messages[0]["content"])
            return await super().chat(messages, tools)

    agent = Agent(
        CapturingProvider([Response(text="here you go")]),
        Settings(model_provider="mock"),
        skills=lib,
    )
    await agent.run("please research quantum computing")
    assert "web-research" in str(captured["system"])
    assert "Search carefully" in str(captured["system"])


async def test_skill_is_authored_after_a_tool_using_task(tmp_path):
    lib = SkillLibrary(tmp_path)
    authored = (
        "---\nname: search-and-report\ndescription: Search the web and report findings\n"
        "triggers: [search, report]\ntags: [web]\n---\n## Steps\n1. Search.\n2. Report."
    )
    scripted = [
        Response(tool_calls=[ToolCall(name="web_search", arguments={"q": "x"}, id="c0")]),
        Response(text="Here is what I found."),
        Response(text=authored),  # the post-task authoring reply
    ]
    agent = Agent(
        MockProvider(scripted),
        Settings(model_provider="mock"),
        tools=_FakeTools(),
        skills=lib,
    )
    await agent.run("search the web and tell me about X")
    assert lib.get("search-and-report") is not None


async def test_no_skill_authored_without_tool_use(tmp_path):
    lib = SkillLibrary(tmp_path)
    agent = Agent(
        MockProvider([Response(text="just a plain answer")]),
        Settings(model_provider="mock"),
        skills=lib,
    )
    await agent.run("what is 2 + 2?")
    assert lib.count() == 0
