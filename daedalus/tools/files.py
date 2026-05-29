"""File tools: read, write, and patch text files inside the workspace."""

from __future__ import annotations

from ._common import safe_path
from .registry import Tool

_MAX_READ = 100_000  # characters; keeps a huge file from blowing up the context


def files_read(path: str) -> str:
    p = safe_path(path)
    if not p.exists():
        return f"ERROR: no such file: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_READ:
        return text[:_MAX_READ] + f"\n... (truncated at {_MAX_READ} chars)"
    return text


def files_write(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def files_patch(path: str, find: str, replace: str) -> str:
    """Replace the first occurrence of ``find`` with ``replace`` in a file."""
    p = safe_path(path)
    if not p.exists():
        return f"ERROR: no such file: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if find not in text:
        return f"ERROR: 'find' text not present in {path}; no change made"
    p.write_text(text.replace(find, replace, 1), encoding="utf-8")
    return f"patched {path}"


TOOLS = [
    Tool(
        name="files_read",
        description="Read a UTF-8 text file from the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "relative path in the workspace"}
            },
            "required": ["path"],
        },
        func=files_read,
    ),
    Tool(
        name="files_write",
        description="Create or overwrite a text file in the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "relative path in the workspace"},
                "content": {"type": "string", "description": "full file contents"},
            },
            "required": ["path", "content"],
        },
        func=files_write,
        requires_approval=True,  # writes are gated once guardrails land (M6)
    ),
    Tool(
        name="files_patch",
        description="Replace the first occurrence of a substring in a workspace file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string", "description": "exact text to find"},
                "replace": {"type": "string", "description": "replacement text"},
            },
            "required": ["path", "find", "replace"],
        },
        func=files_patch,
        requires_approval=True,
    ),
]
