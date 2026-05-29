"""The human-editable memory files: SOUL.md, MEMORY.md and USER.md.

These three Markdown files live in ``~/.dae`` and are loaded into the system prompt
*every* turn, so anything written here is always in the agent's awareness:

  * ``SOUL.md``   — the agent's persona and operating principles (who it *is*).
  * ``MEMORY.md`` — durable world/environment facts ("the project lives in ~/code").
  * ``USER.md``   — who the user is and how they like things done.

Keeping them as plain Markdown means the user can open and edit them by hand — a
persona and memory you can read and correct, not a black box. Edit ``SOUL.md`` to
retune the agent's voice and values; it takes effect on the very next turn.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings

_SOUL_SEED = """\
# Daedalus — Soul

This file is my persona and operating contract. It is loaded into my system prompt on
every turn, so editing it changes how I behave starting with your next message. Make it
yours.

## Identity
I am Daedalus, a helpful, precise, self-hosted AI agent that runs on your own machine.
I act on your requests — answering directly when I can, and reaching for tools when a
task needs the outside world. I am request-scoped: I do the work that was asked, then
stop. I never invent goals of my own.

## Operating principles
- Be concise and concrete. Prefer doing over describing.
- If a request is ambiguous in a way that matters, ask before acting.
- For risky or destructive actions, explain what I'll do and seek confirmation.
- Never fabricate tool results or facts. If I don't know, I say so or look it up.
- Respect the user's machine: stay within the workspace, honour the budget and the
  kill switch, and leave an honest trace of what I did.
"""

_MEMORY_SEED = """\
# Daedalus Memory

Durable facts about the world and environment Daedalus operates in. Daedalus appends
to this as it learns; you can edit it freely.

(Nothing yet.)
"""

_USER_SEED = """\
# About the User

Preferences and profile details Daedalus should always keep in mind.

(Nothing yet — tell Daedalus about yourself and it will remember.)
"""


def _path(name: str) -> Path:
    return get_settings().ensure_home() / name


def ensure_seeded() -> None:
    """Create starter SOUL.md / MEMORY.md / USER.md on first run (never overwrites)."""
    for name, seed in (
        ("SOUL.md", _SOUL_SEED),
        ("MEMORY.md", _MEMORY_SEED),
        ("USER.md", _USER_SEED),
    ):
        p = _path(name)
        if not p.exists():
            p.write_text(seed, encoding="utf-8")


def read_soul_md() -> str:
    p = _path("SOUL.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_memory_md() -> str:
    p = _path("MEMORY.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_user_md() -> str:
    p = _path("USER.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append_memory_md(line: str) -> None:
    p = _path("MEMORY.md")
    existing = p.read_text(encoding="utf-8") if p.exists() else _MEMORY_SEED
    p.write_text(existing.rstrip() + "\n- " + line.strip() + "\n", encoding="utf-8")
