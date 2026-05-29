"""Shell tool: run a command inside the workspace.

Two execution paths, picked at call time:

  * **subprocess** (default) — the command runs through the platform shell in the
    workspace directory with a timeout. Works everywhere, including this Docker-less
    Windows host. The "sandbox" here is the scoped cwd + timeout: honest about its
    limits, not true isolation.
  * **docker** (opt-in via ``SANDBOX=docker``) — the command runs inside a throwaway
    container with the workspace mounted at ``/workspace`` and (by default) no network,
    so a command can't touch the rest of your machine or phone home. If Docker turns
    out not to be installed or running, we **fall back to the subprocess path** rather
    than failing — the agent stays useful on machines without Docker.

This tool carries ``requires_approval=True``; the guardrails (see
:mod:`daedalus.safety.guardrails`) decide whether a given call may proceed.
"""

from __future__ import annotations

import asyncio
import shutil

from ..config import get_settings
from ._common import workspace_dir
from .registry import Tool

_MAX_OUTPUT = 20_000  # characters of combined output to return

# Cache the Docker probe: the answer doesn't change within a process and `docker info`
# is slow enough that we don't want to pay for it on every shell call.
_docker_ok: bool | None = None


async def _docker_available() -> bool:
    """True if the ``docker`` CLI is present *and* the daemon answers. Cached."""
    global _docker_ok
    if _docker_ok is not None:
        return _docker_ok
    if shutil.which("docker") is None:
        _docker_ok = False
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _docker_ok = proc.returncode == 0
    except Exception:  # noqa: BLE001 - any failure means "treat Docker as unavailable"
        _docker_ok = False
    return _docker_ok


async def _drain(proc: asyncio.subprocess.Process, timeout: int) -> str:
    """Collect a started process's output under a timeout and format it for the model."""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return f"ERROR: command timed out after {timeout}s"

    text = (out or b"").decode("utf-8", errors="replace")
    if len(text) > _MAX_OUTPUT:
        text = text[:_MAX_OUTPUT] + f"\n... (truncated at {_MAX_OUTPUT} chars)"
    return f"(exit {proc.returncode})\n{text}".strip()


async def _run_subprocess(command: str, timeout: int) -> str:
    """Run the command through the platform shell, scoped to the workspace dir."""
    cwd = workspace_dir()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: could not start command: {exc}"
    return await _drain(proc, timeout)


async def _run_in_docker(command: str, timeout: int, image: str, network: bool) -> str:
    """Run the command inside a throwaway container with the workspace mounted.

    ``--rm`` cleans up the container; ``--network none`` (the default) cuts egress.
    On failure to launch we return an error string and the caller has already decided
    to use Docker, so we don't silently fall back here — the top-level chooser does.
    """
    cwd = workspace_dir()
    args = ["docker", "run", "--rm", "-v", f"{cwd}:/workspace", "-w", "/workspace"]
    if not network:
        args += ["--network", "none"]
    args += [image, "sh", "-c", command]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: could not start docker ({exc}); is the daemon running?"
    return await _drain(proc, timeout)


async def shell_exec(command: str, timeout: int = 30) -> str:
    """Run a shell command in the workspace, using Docker when configured + available."""
    settings = get_settings()
    if settings.sandbox == "docker" and await _docker_available():
        return await _run_in_docker(
            command, timeout, settings.sandbox_image, settings.sandbox_network
        )
    return await _run_subprocess(command, timeout)


TOOL = Tool(
    name="shell_exec",
    description="Run a shell command inside the workspace and return its combined "
    "stdout/stderr and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "the command line to run"},
            "timeout": {"type": "integer", "description": "max seconds to wait (default 30)"},
        },
        "required": ["command"],
    },
    func=shell_exec,
    requires_approval=True,  # enforced by guardrails (daedalus.safety.guardrails)
)
