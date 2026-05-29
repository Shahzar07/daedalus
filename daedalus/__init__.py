"""Daedalus — a self-hosted, teachable AI agent.

The package is organized so that each concern lives in its own subpackage and the
agent loop (``daedalus.core.loop``) stays the readable centerpiece:

    core/      the ReAct loop, the LLM provider abstraction, context, event bus
    tools/     pluggable tools, auto-discovered from a registry (M2)
    memory/    SQLite/FTS5 recall + MEMORY.md / USER.md (M3)
    surfaces/  terminal (TUI), web, Telegram, WhatsApp (M4+)

Nothing here invents its own goals; Daedalus only runs work the user asks for.
"""

__version__ = "0.1.0"
