"""Turn a pasted ``curl`` command into provider connection settings.

The fastest way to wire Daedalus to *any* OpenAI-compatible endpoint is to copy
the ``curl`` the provider already gave you in their docs and paste it in. This
module pulls the three things we need out of that command:

  * **base_url** — the POST URL with the ``/chat/completions`` suffix stripped, so
    ``https://host/v1/chat/completions`` becomes ``https://host/v1``.
  * **api_key**  — the token from an ``Authorization: Bearer <key>`` header (or
    ``-u user:pass`` basic auth, or an ``x-api-key`` header).
  * **model**    — the ``"model"`` field from the ``-d`` JSON body.

It is deliberately forgiving: line-continuations (``\\``, ``^``, backtick),
single/double quotes, ``--data-raw``/``--data-binary`` aliases, and a missing or
malformed JSON body are all tolerated. Anything it can't find comes back as
``None`` so the caller can fall back to defaults or prompt for it.

Used by :class:`daedalus.config.Settings` (paste a curl into ``CUSTOM_CURL``) and by
the ``dae connect`` CLI command (which parses then writes clean fields to ``.env``).
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

# URL path suffixes we strip to recover the OpenAI-compatible *base* URL. Order
# matters: longest/most-specific first so "/chat/completions" wins over "/completions".
_API_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/responses",
    "/embeddings",
    "/messages",
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_MODEL_RE = re.compile(r'"model"\s*:\s*"([^"]+)"')
_BEARER_RE = re.compile(r"bearer\s+(.+)", re.IGNORECASE)


@dataclass(slots=True)
class CurlConfig:
    """What we managed to extract from a curl command. Any field may be ``None``."""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    def is_empty(self) -> bool:
        return not (self.base_url or self.api_key or self.model)


def _normalize(curl: str) -> str:
    """Flatten shell line-continuations so the whole command is one line.

    Handles POSIX ``\\``-newline, Windows ``^``-newline and PowerShell backtick
    continuations, then collapses any remaining newlines to spaces.
    """
    text = curl.strip()
    # Strip a leading "curl" (case-insensitive) if present.
    text = re.sub(r"^\s*curl\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\\r?\n", " ", text)  # POSIX  backslash-continuation
    text = re.sub(r"\^\r?\n", " ", text)  # cmd.exe caret-continuation
    text = re.sub(r"`\r?\n", " ", text)  # PowerShell backtick-continuation
    text = text.replace("\r", " ").replace("\n", " ")
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Split into shell tokens, tolerating unbalanced quotes (don't raise)."""
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        # Unbalanced quote somewhere — fall back to a naive split so we still try.
        return text.split()


def _strip_suffix(url: str) -> str:
    """Drop a trailing API path (``/chat/completions`` etc.) and any trailing slash."""
    url = url.rstrip("/")
    low = url.lower()
    for suffix in _API_SUFFIXES:
        if low.endswith(suffix):
            return url[: -len(suffix)]
    return url


def parse_curl(curl: str) -> CurlConfig:
    """Best-effort extraction of base_url / api_key / model from a curl command."""
    if not curl or not curl.strip():
        return CurlConfig()

    tokens = _tokenize(_normalize(curl))

    url: str | None = None
    api_key: str | None = None
    body: str | None = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        flag = tok.lower()

        # Headers: -H / --header  "Name: value"
        if flag in ("-h", "--header") and i + 1 < len(tokens):
            name, _, value = tokens[i + 1].partition(":")
            name, value = name.strip().lower(), value.strip()
            if name == "authorization":
                m = _BEARER_RE.match(value)
                api_key = m.group(1).strip() if m else value or None
            elif name in ("x-api-key", "api-key") and value:
                api_key = value
            i += 2
            continue

        # Request body: -d / --data / --data-raw / --data-binary / --data-ascii
        if flag in ("-d", "--data", "--data-raw", "--data-binary", "--data-ascii"):
            if i + 1 < len(tokens):
                body = tokens[i + 1]
            i += 2
            continue

        # Basic auth: -u / --user  user:pass  (treat the whole thing as the key)
        if flag in ("-u", "--user") and i + 1 < len(tokens):
            api_key = api_key or tokens[i + 1]
            i += 2
            continue

        # Flags that take an argument we don't care about — skip the value too.
        if flag in ("-x", "--request", "-a", "--user-agent", "-e", "--referer", "--url"):
            # --url's value *is* the URL; capture it.
            if flag == "--url" and i + 1 < len(tokens):
                url = url or tokens[i + 1]
            i += 2
            continue

        # A bare token that looks like a URL is the endpoint.
        if _URL_RE.match(tok):
            url = url or tok
            i += 1
            continue

        i += 1

    base_url = _strip_suffix(url) if url else None
    model = _model_from_body(body)
    return CurlConfig(base_url=base_url, api_key=api_key, model=model)


def _model_from_body(body: str | None) -> str | None:
    """Pull the ``model`` field out of the JSON request body (regex fallback)."""
    if not body:
        return None
    try:
        obj = json.loads(body)
        if isinstance(obj, dict) and isinstance(obj.get("model"), str):
            return obj["model"]
    except (json.JSONDecodeError, TypeError):
        pass
    m = _MODEL_RE.search(body)
    return m.group(1) if m else None
