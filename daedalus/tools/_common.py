"""Shared helpers for tools that touch the filesystem.

Files and shell commands operate inside a single **workspace** directory
(``~/.dae/workspace`` by default) so the agent can't wander the whole disk. This is
a soft boundary for M2 — real isolation (Docker) arrives in M6 — but path-escape
checks already prevent ``../`` traversal out of the workspace.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings


def workspace_dir() -> Path:
    ws = get_settings().dae_home / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def safe_path(path: str) -> Path:
    """Resolve ``path`` inside the workspace, rejecting anything that escapes it."""
    ws = workspace_dir().resolve()
    candidate = (ws / path).resolve()
    if candidate != ws and ws not in candidate.parents:
        raise ValueError(f"path '{path}' escapes the workspace")
    return candidate
