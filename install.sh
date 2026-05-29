#!/usr/bin/env bash
# Daedalus one-command setup for macOS / Linux.
#   Usage:  ./install.sh
# Installs `dae` as a global command, seeds a .env, and prepares a dev venv. Idempotent.
set -euo pipefail

echo "==> Installing Daedalus (dae)"

# 1. uv is our installer/runner. It manages the venv and Python for us.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but not found."
    echo "Install it from https://docs.astral.sh/uv/ then re-run ./install.sh"
    exit 1
fi

# 2. Sync core dependencies into a project venv (.venv) — used for tests/dev.
echo "==> Syncing dependencies (core)..."
uv sync

# 3. Install `dae` as a GLOBAL command so it works from any shell (like a normal CLI).
#    --editable means it tracks this folder, so `git pull` updates the command too.
echo "==> Installing the global 'dae' command..."
uv tool install --editable .
uv tool update-shell   # ensure uv's tool bin dir is on PATH (idempotent)

# 4. Seed config from the documented template (never overwrite an existing .env).
if [ ! -f .env ]; then
    cp .env.example .env
    echo "==> Wrote .env (defaults to local Ollama, no API keys)."
else
    echo "==> .env already exists; leaving it untouched."
fi

echo ""
echo "Daedalus is installed. Open a NEW terminal, then just run:"
echo "  dae                 # full-screen terminal UI"
echo "  dae --plain         # minimal REPL"
echo "  dae --help          # all commands"
echo ""
echo "Pick a model (edit .env), or connect any endpoint instantly:"
echo "  - Free local:  install Ollama (https://ollama.com), run: ollama pull qwen2.5"
echo "  - Hosted:      set MODEL_PROVIDER + key in .env (Groq/Gemini/OpenRouter/OpenAI)"
echo "  - Any API:     dae connect 'curl https://.../v1/chat/completions -H \"Authorization: Bearer KEY\" -d ...'"
echo ""
echo "Add optional capabilities to the global command (re-run with extras):"
echo '  uv tool install --editable ".[web,telegram,voice,scheduler,semantic,mcp]"'
echo "  # web=Trace Viewer (dae serve)  telegram=bot  voice=mic  scheduler=jobs"
