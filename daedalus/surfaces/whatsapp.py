"""The WhatsApp surface — reach Daedalus on WhatsApp via an unofficial bridge.

WhatsApp has no official self-hosted bot API, so this surface drives a small **Node
companion process** (``bridge/whatsapp``) built on the community ``whatsapp-web.js``
library. That bridge logs in by showing a QR code you scan with your phone (exactly like
WhatsApp Web) and exposes a tiny localhost HTTP API:

    GET  /status            -> {"ready": bool, "qr": "<string|null>"}
    GET  /messages          -> {"messages": [{"from": "...", "body": "..."}]}  (drains a queue)
    POST /send  {to, body}  -> delivers a reply

This Python side spawns that bridge, waits for login, then polls for inbound messages
and feeds each to the **same agent** the terminal uses (with per-sender history). No new
Python dependency — we talk to the bridge with the ``httpx`` we already ship.

**Honest caveats (read before relying on this):** ``whatsapp-web.js`` is an unofficial,
reverse-engineered automation of WhatsApp Web. It can break when WhatsApp changes things,
automating a personal account carries a ban risk, and it needs Node.js installed plus a
phone scan to start. It's here because the brief asks for WhatsApp reach; treat it as the
least stable surface and prefer Telegram for anything you depend on.

Requires Node.js (checked at runtime). See ``bridge/whatsapp/README.md`` for setup.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
from pathlib import Path

import httpx

from ..config import Settings
from ..core.loop import Agent


def _bridge_dir() -> Path:
    """Locate the bundled Node bridge (``<repo>/bridge/whatsapp``)."""
    # daedalus/surfaces/whatsapp.py -> parents[2] is the project root.
    return Path(__file__).resolve().parents[2] / "bridge" / "whatsapp"


class WhatsAppBridge:
    """Manages the Node companion process and bridges its messages to the agent."""

    def __init__(self, agent: Agent, settings: Settings):
        self.agent = agent
        self.settings = settings
        self.base_url = settings.whatsapp_bridge_url.rstrip("/")
        self.run_lock = asyncio.Lock()
        self.histories: dict[str, list] = {}
        self._proc: subprocess.Popen | None = None

    # ---- the Node companion --------------------------------------------------

    def _spawn_bridge(self) -> None:
        """Start ``node index.js`` in the bridge dir, inheriting stdout (it prints the QR)."""
        bridge = _bridge_dir()
        if not (bridge / "index.js").exists():
            raise SystemExit(
                f"WhatsApp bridge not found at {bridge}. See bridge/whatsapp/README.md."
            )
        if not (bridge / "node_modules").exists():
            print("[whatsapp] installing bridge dependencies (one-time: npm install)…")
            npm = shutil.which("npm")
            if npm is None:
                raise SystemExit("npm not found. Install Node.js (which includes npm) and retry.")
            subprocess.run([npm, "install"], cwd=str(bridge), check=True)
        env_port = self.base_url.rsplit(":", 1)[-1]
        self._proc = subprocess.Popen(
            [shutil.which("node") or "node", "index.js"],
            cwd=str(bridge),
            env={"PORT": env_port, "PATH": __import__("os").environ.get("PATH", "")},
        )

    def _stop_bridge(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)

    # ---- bridging messages to the agent -------------------------------------

    async def reply_to(self, sender: str, text: str) -> str:
        """Run one inbound message through the agent with this sender's own history."""
        async with self.run_lock:
            self.agent.history = self.histories.get(sender, [])
            try:
                answer = await self.agent.run(text)
            except Exception as exc:  # noqa: BLE001 - report, never crash the loop
                answer = f"error: {exc}"
            finally:
                self.histories[sender] = self.agent.history
        return answer

    async def _wait_until_ready(self, client: httpx.AsyncClient) -> None:
        """Block until the bridge reports a logged-in session, surfacing the QR once."""
        shown_qr = False
        while True:
            try:
                status = (await client.get(f"{self.base_url}/status", timeout=10)).json()
            except Exception:  # noqa: BLE001 - bridge still booting; keep waiting
                await asyncio.sleep(1.0)
                continue
            if status.get("ready"):
                print("[whatsapp] logged in — bridge is live.")
                return
            if status.get("qr") and not shown_qr:
                print(
                    "[whatsapp] scan the QR code shown in this terminal with your phone "
                    "(WhatsApp > Linked Devices > Link a Device)."
                )
                shown_qr = True
            await asyncio.sleep(1.0)

    async def _poll_loop(self, client: httpx.AsyncClient) -> None:
        """Drain inbound messages and reply, forever (until cancelled)."""
        while True:
            try:
                resp = await client.get(f"{self.base_url}/messages", timeout=30)
                messages = resp.json().get("messages", [])
            except Exception:  # noqa: BLE001 - transient bridge hiccup; back off and retry
                await asyncio.sleep(1.0)
                continue
            for msg in messages:
                sender, body = msg.get("from", ""), (msg.get("body") or "").strip()
                if not sender or not body:
                    continue
                answer = await self.reply_to(sender, body)
                with contextlib.suppress(Exception):
                    await client.post(
                        f"{self.base_url}/send", json={"to": sender, "body": answer}, timeout=30
                    )
            await asyncio.sleep(0.5)

    async def serve(self) -> None:
        """Spawn the bridge, wait for login, then bridge messages until interrupted."""
        self._spawn_bridge()
        try:
            async with httpx.AsyncClient() as client:
                await self._wait_until_ready(client)
                await self._poll_loop(client)
        finally:
            self._stop_bridge()


def run_whatsapp(agent: Agent, settings: Settings) -> None:
    """Entry point for ``dae whatsapp``. Checks for Node, then runs the bridge loop."""
    if shutil.which("node") is None:
        raise SystemExit(
            "Node.js is required for the WhatsApp bridge but was not found.\n"
            "Install it from https://nodejs.org/ (LTS), then run `dae whatsapp` again.\n"
            "Prefer something more stable? Use the Telegram surface (`dae telegram`)."
        )
    bridge = WhatsAppBridge(agent, settings)
    print("[whatsapp] starting the unofficial bridge (Ctrl-C to stop)…")
    try:
        asyncio.run(bridge.serve())
    except KeyboardInterrupt:
        print("\n[whatsapp] stopped.")
