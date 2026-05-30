"""Provider resolution: the one bit of config logic worth pinning down."""

from daedalus.config import Settings


def test_hosted_provider_resolves_url_and_key():
    s = Settings(model_provider="groq", groq_api_key="k")
    assert s.resolved_base_url().startswith("https://api.groq.com")
    assert s.resolved_api_key() == "k"


def test_ollama_is_keyless_and_local():
    s = Settings(model_provider="ollama")
    assert "11434" in s.resolved_base_url()
    assert s.resolved_api_key() is None


def test_gemini_uses_openai_compat_endpoint():
    s = Settings(model_provider="gemini", gemini_api_key="g")
    assert "generativelanguage.googleapis.com" in s.resolved_base_url()
    assert s.resolved_api_key() == "g"


def test_custom_without_base_url_is_none():
    # The provider raises a helpful error; config just reports "unset".
    assert Settings(model_provider="custom").resolved_base_url() is None


def test_explicit_base_url_overrides_default():
    s = Settings(model_provider="ollama", openai_base_url="http://host:9/v1")
    assert s.resolved_base_url() == "http://host:9/v1"


# ---- hardening: a corrupted OPENAI_BASE_URL must never reach the HTTP client ----
# This is the exact crash a user hit: a multi-line curl pasted into a single-line
# prompt left a stray header fragment in OPENAI_BASE_URL, and httpx died with
# "Request URL is missing an 'http://' or 'https://' protocol".


def test_schemeless_base_url_is_ignored_falls_back_to_provider_default():
    # gemini with a garbage OPENAI_BASE_URL (a leftover curl header) must fall back to
    # the gemini default, not hand the garbage to httpx.
    s = Settings(
        model_provider="gemini",
        gemini_api_key="g",
        openai_base_url='-H "x-goog-api-key: AIza..."',
    )
    assert "generativelanguage.googleapis.com" in s.resolved_base_url()


def test_schemeless_base_url_on_custom_resolves_to_none():
    # custom has no built-in default; a garbage URL must resolve to None (the provider
    # then raises a clear "set OPENAI_BASE_URL" error) rather than a mid-request crash.
    s = Settings(model_provider="custom", openai_base_url="not-a-url")
    assert s.resolved_base_url() is None


def test_curl_base_url_must_have_scheme_too():
    # A base URL parsed from a pasted curl is held to the same standard.
    s = Settings(model_provider="custom", custom_curl="curl ftp://host/v1 -d {}")
    assert s.resolved_base_url() is None


def test_valid_base_url_is_trimmed():
    s = Settings(model_provider="custom", openai_base_url="  https://host/v1  ")
    assert s.resolved_base_url() == "https://host/v1"
