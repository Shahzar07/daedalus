"""Builds what the model sees each turn.

Right now that's a system prompt (identity + operating rules + how to call tools)
plus the running conversation. In M3 this same module grows to inject recalled
memories, ``MEMORY.md`` and ``USER.md`` — which is exactly why context assembly is
isolated here instead of being tangled into the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .llm import Message

if TYPE_CHECKING:
    from ..skills.engine import Skill

_IDENTITY = """\
You are Daedalus, a helpful, precise, self-hosted AI agent that runs on the user's \
own machine. You act on the user's requests — answering directly when you can, and \
using tools when a task needs the outside world. You are request-scoped: you do the \
work that was asked, then stop. You never invent goals of your own.

Operating principles:
- Be concise and concrete. Prefer doing over describing.
- If a request is ambiguous in a way that matters, ask before acting.
- For risky or destructive actions, explain what you'll do and seek confirmation.
- Never fabricate tool results or facts. If you don't know, say so or look it up.\
"""

_TOOL_INSTRUCTIONS = """\

You have tools available. Prefer calling them through the normal function-calling \
interface. If — and only if — you cannot emit a native tool call, you may instead \
reply with a single fenced block in this exact form:

```action
{{"tool": "<tool_name>", "args": {{ ... }}}}
```

After a tool runs you'll receive its result; use it to continue. When you have \
enough to answer, reply normally with the final answer and no action block.

Available tools: {tool_list}\
"""


def _section(title: str, body: str) -> str:
    body = body.strip()
    return f"\n\n## {title}\n{body}" if body else ""


def _render_skills(skills: list[Skill]) -> str:
    """Format matched skills as a single prompt section the model can follow."""
    intro = (
        "These saved playbooks match the request. Follow the most relevant one when it "
        "applies; adapt freely if the situation differs.\n"
    )
    return intro + "\n\n".join(s.to_prompt_block() for s in skills)


def build_system_prompt(
    tool_names: list[str] | None = None,
    memory_md: str = "",
    user_md: str = "",
    recalled: list[str] | None = None,
    skills: list[Skill] | None = None,
    soul_md: str = "",
) -> str:
    """Assemble the system prompt: identity, tools, matched skills, then any memory.

    The persona comes from ``SOUL.md`` when present (user-editable, loaded every turn);
    otherwise the built-in ``_IDENTITY`` is used so the agent always has a voice. Tool
    guidance appears only when tools exist; skills and memory sections appear only when
    there's something to show. That keeps the prompt as small as the situation allows.
    """
    prompt = soul_md.strip() if soul_md and soul_md.strip() else _IDENTITY
    if tool_names:
        prompt += _TOOL_INSTRUCTIONS.format(tool_list=", ".join(tool_names))
    if skills:
        prompt += _section("Relevant skills (playbooks)", _render_skills(skills))
    prompt += _section("About the user (USER.md)", user_md)
    prompt += _section("What you know (MEMORY.md)", memory_md)
    if recalled:
        bullets = "\n".join(f"- {m}" for m in recalled)
        prompt += _section("Possibly relevant memories", bullets)
    return prompt


def assemble_messages(
    history: list[Message],
    user_input: str,
    tool_names: list[str] | None = None,
    memory_md: str = "",
    user_md: str = "",
    recalled: list[str] | None = None,
    skills: list[Skill] | None = None,
    soul_md: str = "",
) -> list[Message]:
    """Compose the message list for a turn: system prompt, prior turns, new input.

    ``history`` is the transcript of earlier exchanges (already including any tool
    calls/results). The system prompt is rebuilt every turn so it always reflects the
    currently available tools, freshly matched skills, recalled memory, and persona.
    """
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                tool_names, memory_md, user_md, recalled, skills, soul_md
            ),
        },
        *history,
        {"role": "user", "content": user_input},
    ]
