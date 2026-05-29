"""Typed configuration — the one piece of global state Daedalus allows.

Why this exists: every subsystem (providers, loop, memory, tools) needs a few
settings, and we want them validated once, in one place, sourced from ``.env`` or
the real environment. Using ``pydantic-settings`` means a bad value fails loudly at
startup instead of mysteriously deep in a request.

The key trick lives in :meth:`Settings.resolved_base_url` / ``resolved_api_key``:
every supported provider speaks the OpenAI-compatible HTTP API, so we map the
chosen provider to a base URL + key and let a single client (``core/llm.py``)
handle them all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.curl import CurlConfig, parse_curl

Provider = Literal["ollama", "groq", "gemini", "openrouter", "openai", "custom", "mock"]

# provider -> (default OpenAI-compatible base URL, name of the key field on Settings).
# A base URL of None for "custom" means the user MUST supply OPENAI_BASE_URL.
# "mock" needs neither — it runs entirely in-process.
_PROVIDER_DEFAULTS: dict[str, tuple[str | None, str | None]] = {
    "ollama": ("http://localhost:11434/v1", None),
    "groq": ("https://api.groq.com/openai/v1", "groq_api_key"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_api_key"),
    "openai": ("https://api.openai.com/v1", "openai_api_key"),
    "custom": (None, "openai_api_key"),
    "mock": (None, None),
}


class Settings(BaseSettings):
    # protected_namespaces=() lets us use the natural names model_provider /
    # model_name without pydantic complaining about its reserved "model_" prefix.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # --- model selection ---
    model_provider: Provider = "ollama"
    model_name: str = "qwen2.5"

    # --- provider keys (only the selected provider's key is needed) ---
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    # --- custom / override endpoint ---
    openai_base_url: str = ""

    # Paste a whole `curl ...` command here (or use `dae connect`) and Daedalus
    # extracts the base URL, Bearer key, and model from it. Explicit fields above
    # always win over anything parsed from the curl.
    custom_curl: str = ""

    # --- agent limits & safety ---
    max_steps: int = 12
    daily_budget_usd: float = 1.00

    # How the shell tool runs commands:
    #   subprocess - run in a scoped working dir with a timeout (default; works
    #                everywhere, including this Docker-less host)
    #   docker     - run inside a throwaway container for real isolation; if Docker
    #                isn't actually available at runtime, the tool falls back to
    #                subprocess automatically rather than failing.
    sandbox: Literal["docker", "subprocess"] = "subprocess"
    # Used only when sandbox="docker": the image to run in, and whether the container
    # gets network access (off by default so a sandboxed command can't phone home).
    sandbox_image: str = "python:3.11-slim"
    sandbox_network: bool = False

    # How tools flagged ``requires_approval`` (shell, file writes) are handled:
    #   ask   - prompt the user interactively (default; safest for a human at a surface)
    #   auto  - run them without asking (use only when you trust the workload)
    #   deny  - never run them
    approval_policy: Literal["ask", "auto", "deny"] = "ask"

    # Price used to estimate spend against ``daily_budget_usd``. 0 = free (local
    # Ollama, free tiers), so the cap never trips. Set it for paid providers, e.g.
    # ~0.0006 for a cheap hosted model, to make the daily cap meaningful.
    cost_per_1k_tokens: float = 0.0

    # Per-request model timeout (seconds). Generous by default: large hosted models
    # under load and CPU-only local models can take a while to produce a first token.
    request_timeout: float = 180.0

    # --- M10 upgrades ---
    # Reflection: a second model pass critiques (and may rewrite) each answer. Off by
    # default because it doubles the model calls per turn; great for accuracy when on.
    reflection: bool = False
    reflection_max_revisions: int = 1
    # Subagents: let the agent delegate self-contained subtasks to bounded child agents.
    # Cheap and on by default; hard-capped on depth and per-request count.
    subagents: bool = True
    subagent_max_depth: int = 1
    subagent_max_children: int = 3
    # Memory graph: lightweight entity/relation extraction over remembered facts ($0,
    # no extra deps) so recall can surface *connected* knowledge, not just keyword hits.
    memory_graph: bool = True
    # Semantic memory: recall by meaning via sentence-transformers. Needs the [semantic]
    # extra and a one-time model download, so it's off until you opt in.
    semantic_memory: bool = False

    # --- voice (needs the [voice] extra; used by `dae voice`) ---
    whisper_model: str = "base"  # faster-whisper size: tiny|base|small|medium|large-v3
    piper_voice: str = ""  # path to a downloaded Piper voice (.onnx) for text-to-speech

    # --- tools ---
    searxng_url: str = ""

    # --- surfaces / integrations ---
    telegram_bot_token: str = ""
    # Comma-separated chat IDs allowed to talk to the bot (e.g. "123456789,987654321").
    # Empty = "open mode": the bot answers anyone and warns loudly at startup. Setting
    # this is the recommended security boundary for a public bot username.
    telegram_allowed_chat_ids: str = ""
    # The WhatsApp surface talks to a small local Node bridge (unofficial; see
    # bridge/whatsapp). This is where that bridge's HTTP API listens.
    whatsapp_bridge_url: str = "http://127.0.0.1:8765"
    mcp_servers: list[dict[str, Any]] = []

    # --- paths ---
    dae_home: Path = Path.home() / ".dae"

    # ---- derived helpers -----------------------------------------------------

    def parsed_curl(self) -> CurlConfig:
        """Connection settings extracted from ``custom_curl`` (empty if unset)."""
        return parse_curl(self.custom_curl) if self.custom_curl.strip() else CurlConfig()

    def resolved_base_url(self) -> str | None:
        """Effective base URL for the active provider.

        Precedence: an explicit ``OPENAI_BASE_URL`` wins (handy for a non-default
        Ollama host or a proxy), then a base URL parsed from a pasted ``CUSTOM_CURL``,
        then the provider's built-in default.
        """
        if self.openai_base_url:
            return self.openai_base_url
        curl = self.parsed_curl()
        if curl.base_url:
            return curl.base_url
        return _PROVIDER_DEFAULTS[self.model_provider][0]

    def resolved_api_key(self) -> str | None:
        """API key for the active provider, or None for keyless providers.

        An explicitly configured key field wins; otherwise we accept a key parsed
        from ``CUSTOM_CURL`` (so a single pasted curl is enough to connect).
        """
        key_field = _PROVIDER_DEFAULTS[self.model_provider][1]
        explicit = getattr(self, key_field) if key_field else None
        if explicit:
            return explicit
        return self.parsed_curl().api_key

    def resolved_model(self) -> str:
        """Model name to send. A model parsed from ``CUSTOM_CURL`` wins, since
        pasting that curl is an explicit statement of which model to call."""
        curl = self.parsed_curl()
        return curl.model or self.model_name

    def telegram_allowlist(self) -> set[int]:
        """Parse ``telegram_allowed_chat_ids`` into a set of ints (empty = open mode)."""
        ids: set[int] = set()
        for part in self.telegram_allowed_chat_ids.replace(" ", "").split(","):
            if part:
                try:
                    ids.add(int(part))
                except ValueError:
                    pass  # ignore a malformed entry rather than failing startup
        return ids

    def ensure_home(self) -> Path:
        """Create ~/.dae (or DAE_HOME) on demand. Components that persist state
        call this rather than doing filesystem work at import time."""
        self.dae_home.mkdir(parents=True, exist_ok=True)
        return self.dae_home


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so the env/.env is read once."""
    return Settings()
