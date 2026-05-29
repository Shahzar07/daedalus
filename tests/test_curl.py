"""Tests for the curl-command parser and its integration into Settings."""

from __future__ import annotations

from daedalus.config import Settings
from daedalus.core.curl import parse_curl

# The exact curl a user might paste (the GLM endpoint used in the live demo).
_REAL = (
    'curl -X POST "https://api.us-west-2.modal.direct/v1/chat/completions" '
    '-H "Content-Type: application/json" '
    '-H "Authorization: Bearer modalresearch_TESTKEY123" '
    '-d \'{"model": "zai-org/GLM-5.1-FP8", "messages": [{"role":"user","content":"hi"}], '
    '"max_tokens": 500}\''
)


def test_parses_url_key_and_model():
    cfg = parse_curl(_REAL)
    assert cfg.base_url == "https://api.us-west-2.modal.direct/v1"
    assert cfg.api_key == "modalresearch_TESTKEY123"
    assert cfg.model == "zai-org/GLM-5.1-FP8"


def test_strips_chat_completions_suffix():
    cfg = parse_curl('curl https://example.com/v1/chat/completions -d \'{"model":"m"}\'')
    assert cfg.base_url == "https://example.com/v1"


def test_handles_backslash_line_continuations():
    multiline = (
        "curl https://host/v1/chat/completions \\\n"
        '  -H "Authorization: Bearer sk-abc" \\\n'
        '  -d \'{"model": "gpt-4o-mini"}\''
    )
    cfg = parse_curl(multiline)
    assert cfg.base_url == "https://host/v1"
    assert cfg.api_key == "sk-abc"
    assert cfg.model == "gpt-4o-mini"


def test_accepts_x_api_key_header():
    cfg = parse_curl('curl https://host/v1/chat/completions -H "x-api-key: key-9"')
    assert cfg.api_key == "key-9"


def test_url_via_explicit_flag():
    cfg = parse_curl('curl --url https://host/v1/completions -H "Authorization: Bearer k"')
    assert cfg.base_url == "https://host/v1"


def test_tolerates_malformed_json_body_for_model():
    # Trailing junk makes json.loads fail; the regex fallback still finds the model.
    cfg = parse_curl('curl https://h/v1/chat/completions -d \'{"model": "x-1", oops}\'')
    assert cfg.model == "x-1"


def test_empty_input_is_empty_config():
    cfg = parse_curl("")
    assert cfg.is_empty()
    assert cfg.base_url is None


def test_no_url_returns_none_base_url():
    cfg = parse_curl('curl -H "Authorization: Bearer k" -d \'{"model":"m"}\'')
    assert cfg.base_url is None
    assert cfg.api_key == "k"
    assert cfg.model == "m"


# ---- Settings integration ---------------------------------------------------


def test_settings_resolves_from_custom_curl():
    s = Settings(model_provider="custom", custom_curl=_REAL)
    assert s.resolved_base_url() == "https://api.us-west-2.modal.direct/v1"
    assert s.resolved_api_key() == "modalresearch_TESTKEY123"
    assert s.resolved_model() == "zai-org/GLM-5.1-FP8"


def test_explicit_fields_win_over_curl():
    s = Settings(
        model_provider="custom",
        custom_curl=_REAL,
        openai_base_url="https://override.example/v1",
        openai_api_key="explicit-key",
    )
    assert s.resolved_base_url() == "https://override.example/v1"
    assert s.resolved_api_key() == "explicit-key"
    # No explicit model field, so the curl's model still applies.
    assert s.resolved_model() == "zai-org/GLM-5.1-FP8"


def test_no_curl_falls_back_to_defaults():
    s = Settings(model_provider="ollama")
    assert s.resolved_base_url() == "http://localhost:11434/v1"
    assert s.resolved_api_key() is None
    assert s.resolved_model() == "qwen2.5"
