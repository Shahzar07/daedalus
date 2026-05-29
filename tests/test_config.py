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
