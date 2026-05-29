"""Auto-author skills from successful work.

After the agent finishes a task that actually *did* something (used tools across one
or more steps), this asks the model a narrow question: "was that a reusable
procedure, and if so, write the playbook." If yes, we save a new ``SKILL.md`` so the
next similar request starts from the recipe instead of from scratch.

It is strictly best-effort and conservative — most turns produce nothing, and a
failure here never affects the answer the user already received. The model replies in
the exact SKILL.md format (frontmatter + body) or with the single word ``NONE``.
"""

from __future__ import annotations

from ..core.llm import LLMProvider
from .engine import Skill, SkillLibrary, parse_skill_md

_AUTHOR_SYS = (
    "You curate a library of reusable 'skills' — short Markdown playbooks an AI agent "
    "follows to repeat a task reliably. You are conservative: only propose a skill for "
    "a genuinely reusable, generalizable procedure (not a one-off or a trivial answer). "
    "Never invent steps the agent didn't actually take."
)

_AUTHOR_TEMPLATE = """\
A task just completed successfully. Decide if it represents a reusable procedure
worth saving as a skill.

User request:
{user_input}

Tools the agent used: {tools_used}

Final answer:
{final_text}

Existing skills (do NOT duplicate these): {existing}

If it is NOT worth saving, reply with exactly: NONE

If it IS worth saving, reply with ONLY a SKILL.md document in this exact format:
---
name: short-kebab-case-name
description: one sentence on when to use this skill
triggers: [a few, request, keywords]
tags: [topic]
---
## When to use
<one or two lines>

## Steps
1. <generalized step>
2. <generalized step>

## Notes
<gotchas, optional>
"""


async def maybe_author_skill(
    provider: LLMProvider,
    library: SkillLibrary,
    user_input: str,
    final_text: str,
    tools_used: list[str],
) -> Skill | None:
    """Possibly write a new skill from a just-finished task. Returns it if created.

    Only fires when at least one tool ran — a plain Q&A answer isn't a procedure.
    """
    if not tools_used:
        return None

    existing = ", ".join(s.name for s in library.all()) or "(none yet)"
    prompt = _AUTHOR_TEMPLATE.format(
        user_input=user_input.strip(),
        tools_used=", ".join(sorted(set(tools_used))),
        final_text=final_text.strip()[:1500],
        existing=existing,
    )
    try:
        response = await provider.chat(
            [
                {"role": "system", "content": _AUTHOR_SYS},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception:  # noqa: BLE001 - authoring is best-effort, never fatal
        return None

    text = (response.text or "").strip()
    if not text or text.upper().startswith("NONE") or "---" not in text:
        return None

    skill = parse_skill_md(text)
    # Guard against junk and against silently overwriting an existing skill.
    if skill is None or not skill.description or library.get(skill.name) is not None:
        return None
    library.add(skill)
    return skill
