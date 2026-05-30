"""The setup wizard's hardened bits: multi-line curl paste, model catalog, model picker.

These are the fixes for the reported onboarding failures — a multi-line curl getting
truncated into a bad base URL, and "it's asking for a model name" with no way to see
what the endpoint offers. All hermetic: HTTP is stubbed with ``httpx.MockTransport``
and prompts are monkeypatched, so nothing touches the network or a real terminal.
"""

from __future__ import annotations

import httpx

from daedalus import cli
from daedalus.core.curl import parse_curl

# ---- _read_curl_arg: gather a whole multi-line paste (the corruption fix) --------


def _fake_tty(monkeypatch) -> None:
    class _Stdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdin", _Stdin())


def test_read_curl_arg_gathers_multiple_lines(monkeypatch):
    _fake_tty(monkeypatch)
    pasted = iter(
        [
            "curl https://api.example.com/v1/chat/completions \\",
            '  -H "Authorization: Bearer SECRET" \\',
            '  -d \'{"model": "demo-1"}\'',
            "",  # a blank line ends the paste
        ]
    )
    monkeypatch.setattr(cli.console, "input", lambda *a, **k: next(pasted))

    text = cli._read_curl_arg(None)

    # The whole command survived, not just the first line.
    assert "api.example.com" in text
    assert "Authorization" in text
    assert "demo-1" in text
    # And it parses back to clean fields — proving the header didn't poison the URL.
    cfg = parse_curl(text)
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.api_key == "SECRET"
    assert cfg.model == "demo-1"


def test_read_curl_arg_returns_explicit_argument_unchanged():
    assert cli._read_curl_arg("curl https://h/v1") == "curl https://h/v1"


# ---- _fetch_models: read an endpoint's /models catalog (best-effort) -------------


def _patch_models_http(monkeypatch, handler) -> None:
    """Point cli's httpx.AsyncClient at an in-memory MockTransport."""
    real_client = httpx.AsyncClient  # capture before patching to avoid self-recursion

    def factory(*_a, **_k):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(cli.httpx, "AsyncClient", factory)


async def test_fetch_models_parses_and_strips_gemini_prefix(monkeypatch):
    def handler(_req):
        return httpx.Response(
            200,
            json={"data": [{"id": "models/gemini-2.0-flash"}, {"id": "alpha"}, {"id": "alpha"}]},
        )

    _patch_models_http(monkeypatch, handler)
    out = await cli._fetch_models("https://x/v1", "key")
    # sorted, de-duped, and the Gemini "models/" prefix removed.
    assert out == ["alpha", "gemini-2.0-flash"]


async def test_fetch_models_empty_on_http_error(monkeypatch):
    _patch_models_http(monkeypatch, lambda _req: httpx.Response(500, json={"error": "down"}))
    assert await cli._fetch_models("https://x/v1", None) == []


async def test_fetch_models_empty_without_base_url():
    assert await cli._fetch_models("", None) == []
    assert await cli._fetch_models(None, None) == []


# ---- _choose_model: pick from the catalog, or type any id ------------------------


async def test_choose_model_picks_by_number(monkeypatch):
    async def fake_fetch(_b, _k):
        return ["alpha", "beta", "gamma"]

    monkeypatch.setattr(cli, "_fetch_models", fake_fetch)
    monkeypatch.setattr(cli, "_ask", lambda *a, **k: "2")
    assert await cli._choose_model("https://x/v1", None) == "beta"


async def test_choose_model_floats_recommended_default_to_top(monkeypatch):
    async def fake_fetch(_b, _k):
        return ["a", "b", "rec"]

    monkeypatch.setattr(cli, "_fetch_models", fake_fetch)
    monkeypatch.setattr(cli, "_ask", lambda *a, **k: "1")  # Enter -> #1
    assert await cli._choose_model("https://x/v1", None, default="rec") == "rec"


async def test_choose_model_accepts_typed_id_not_in_list(monkeypatch):
    async def fake_fetch(_b, _k):
        return ["a", "b"]

    monkeypatch.setattr(cli, "_fetch_models", fake_fetch)
    monkeypatch.setattr(cli, "_ask", lambda *a, **k: "zai-org/GLM-5.1-FP8")
    assert await cli._choose_model("https://x/v1", None) == "zai-org/GLM-5.1-FP8"


async def test_choose_model_falls_back_to_text_when_no_catalog(monkeypatch):
    async def fake_fetch(_b, _k):
        return []

    monkeypatch.setattr(cli, "_fetch_models", fake_fetch)
    monkeypatch.setattr(cli, "_ask", lambda *a, **k: "typed-by-hand")
    assert await cli._choose_model("https://x/v1", None, default="d") == "typed-by-hand"


# ---- _configure_provider: switching providers clears stale custom-endpoint vars --


async def test_keyed_provider_clears_stale_custom_endpoint(monkeypatch, tmp_path):
    # Simulate picking Gemini (#3 in the menu), entering a key, declining the live test.
    monkeypatch.setattr(cli, "_choose", lambda *a, **k: 2)  # 0-based index -> "gemini"
    answers = iter(["test-key", "n"])  # API key, then "no" to the connection-test prompt
    monkeypatch.setattr(cli, "_ask", lambda *a, **k: next(answers))

    async def fake_choose_model(*_a, **_k):
        return "gemini-2.0-flash"

    monkeypatch.setattr(cli, "_choose_model", fake_choose_model)

    captured: dict[str, str] = {}
    monkeypatch.setattr(cli, "_write_env", lambda updates, path: captured.update(updates))

    await cli._configure_provider(tmp_path / ".env")

    assert captured["MODEL_PROVIDER"] == "gemini"
    assert captured["GEMINI_API_KEY"] == "test-key"
    assert captured["MODEL_NAME"] == "gemini-2.0-flash"
    # The crux: a leftover custom endpoint from a prior run is wiped so it can't shadow
    # Gemini's own default base URL (which is what caused the original crash).
    assert captured["OPENAI_BASE_URL"] == ""
    assert captured["CUSTOM_CURL"] == ""
