# Daedalus one-command setup for Windows (PowerShell).
#   Usage:  ./install.ps1
# Installs `dae` as a global command, seeds a .env, and prepares a dev venv. Idempotent.

$ErrorActionPreference = "Stop"

Write-Host "==> Installing Daedalus (dae)" -ForegroundColor Cyan

# 1. uv is our installer/runner. It manages the venv and Python for us.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is required but not found." -ForegroundColor Yellow
    Write-Host "Install it from https://docs.astral.sh/uv/ then re-run ./install.ps1"
    exit 1
}

# 2. Sync core dependencies into a project venv (.venv) — used for tests/dev.
Write-Host "==> Syncing dependencies (core)..." -ForegroundColor Cyan
uv sync

# 3. Install `dae` as a GLOBAL command so it works from any shell (like a normal CLI).
#    --editable means it tracks this folder, so `git pull` updates the command too.
Write-Host "==> Installing the global 'dae' command..." -ForegroundColor Cyan
uv tool install --editable .
uv tool update-shell   # ensure uv's tool bin dir is on PATH (idempotent)

# 4. Seed config from the documented template (never overwrite an existing .env).
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Wrote .env (defaults to local Ollama, no API keys)." -ForegroundColor Green
} else {
    Write-Host "==> .env already exists; leaving it untouched." -ForegroundColor Green
}

Write-Host ""
Write-Host "Daedalus is installed. Open a NEW terminal, then just run:" -ForegroundColor Green
Write-Host "  dae                 # full-screen terminal UI"
Write-Host "  dae --plain         # minimal REPL"
Write-Host "  dae --help          # all commands"
Write-Host ""
Write-Host "Pick a model (edit .env), or connect any endpoint instantly:" -ForegroundColor Cyan
Write-Host "  - Free local:  install Ollama (https://ollama.com), run: ollama pull qwen2.5"
Write-Host "  - Hosted:      set MODEL_PROVIDER + key in .env (Groq/Gemini/OpenRouter/OpenAI)"
Write-Host "  - Any API:     dae connect 'curl https://.../v1/chat/completions -H \"Authorization: Bearer KEY\" -d ...'"
Write-Host ""
Write-Host "Add optional capabilities to the global command (re-run with extras):" -ForegroundColor Cyan
Write-Host '  uv tool install --editable ".[web,telegram,voice,scheduler,semantic,mcp]"'
Write-Host "  # web=Trace Viewer (dae serve)  telegram=bot  voice=mic  scheduler=jobs"
