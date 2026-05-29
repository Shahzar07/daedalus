"""The JSON fallback parser, the mock provider, and transient-error retry."""

import httpx
import pytest

from daedalus.core import llm
from daedalus.core.llm import (
    LLMError,
    MockProvider,
    OpenAICompatibleProvider,
    Response,
    _retry_after,
    extract_json_tool_calls,
)


def test_parse_fenced_action_block():
    text = 'Sure.\n```action\n{"tool": "web_search", "args": {"query": "cats"}}\n```'
    calls = extract_json_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "cats"}


def test_parse_openai_style_keys():
    calls = extract_json_tool_calls('{"name": "files_read", "arguments": {"path": "x"}}')
    assert calls[0].name == "files_read"
    assert calls[0].arguments == {"path": "x"}


def test_parse_returns_empty_for_plain_text():
    assert extract_json_tool_calls("just a normal answer") == []
    assert extract_json_tool_calls(None) == []


async def test_mock_provider_echoes_last_user_message():
    resp = await MockProvider().chat([{"role": "user", "content": "hello"}])
    assert resp.text == "(mock) hello"


async def test_mock_provider_returns_script_in_order():
    scripted = [Response(text="first"), Response(text="second")]
    p = MockProvider(scripted)
    assert (await p.chat([])).text == "first"
    assert (await p.chat([])).text == "second"


# ---- transient-error retry (against a mocked OpenAI-compatible endpoint) --------

_OK = {"choices": [{"message": {"content": "ok"}}]}


async def _noop_sleep(*_a, **_k):
    return None


def _provider(handler, max_retries=3):
    """An OpenAICompatibleProvider whose HTTP calls go to an in-memory mock transport."""
    p = OpenAICompatibleProvider("http://x/v1", "m", api_key="k", max_retries=max_retries)
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return p


def test_retry_after_prefers_header_then_falls_back_to_backoff():
    resp = httpx.Response(429, headers={"retry-after": "2"})
    assert _retry_after(resp, attempt=0) == 2.0
    bare = httpx.Response(429)
    assert _retry_after(bare, attempt=0) == 0.5
    assert _retry_after(bare, attempt=3) == 4.0
    assert _retry_after(bare, attempt=10) == 8.0  # capped


async def test_chat_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.asyncio, "sleep", _noop_sleep)
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "busy"})
        return httpx.Response(200, json=_OK)

    p = _provider(handler, max_retries=3)
    resp = await p.chat([{"role": "user", "content": "hi"}])
    assert resp.text == "ok"
    assert calls["n"] == 3  # two 429s, then success
    await p.aclose()


async def test_chat_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm.asyncio, "sleep", _noop_sleep)
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(503, json={"error": "down"})

    p = _provider(handler, max_retries=2)
    with pytest.raises(LLMError) as exc:
        await p.chat([{"role": "user", "content": "hi"}])
    assert "503" in str(exc.value)
    assert calls["n"] == 3  # initial try + 2 retries
    await p.aclose()


async def test_chat_does_not_retry_on_auth_error(monkeypatch):
    monkeypatch.setattr(llm.asyncio, "sleep", _noop_sleep)
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    p = _provider(handler, max_retries=3)
    with pytest.raises(LLMError) as exc:
        await p.chat([{"role": "user", "content": "hi"}])
    assert "401" in str(exc.value)
    assert calls["n"] == 1  # 401 is a config error — surfaced immediately
    await p.aclose()
