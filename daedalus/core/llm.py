"""The model provider abstraction — one interface, every backend.

The whole point of this file: the agent loop should never care *which* model is
behind it. It calls ``provider.chat(messages, tools)`` and gets back a uniform
:class:`Response`. Swapping Ollama for Groq for Gemini is a one-line change in
``.env``.

How we keep that simple: every provider we support (Ollama, Groq, Gemini,
OpenRouter, OpenAI, and any "custom" server) speaks the **OpenAI-compatible**
``POST /chat/completions`` wire format. So a single :class:`OpenAICompatibleProvider`
— a thin httpx wrapper — handles all of them; only the base URL and key differ.
:class:`MockProvider` runs in-process for tests and offline demos.

Tool calling: modern models return structured ``tool_calls``. Smaller local models
often can't, so :func:`extract_json_tool_calls` recovers a tool call from a plain
`````action`` JSON block in the text — the loop tries native first, then this
fallback.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings

# Transient HTTP statuses worth retrying: rate limits (429) and brief upstream
# hiccups (gateway/unavailable/timeout). A 4xx like 401/404 is a config error — we
# surface those immediately rather than hammering the endpoint.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Messages are plain OpenAI-format dicts ({"role": ..., "content": ...}) so there's
# no translation layer between us and the wire. Tool definitions are the OpenAI
# function schema. Keeping these as dicts is deliberately transparent.
Message = dict[str, Any]
ToolSchema = dict[str, Any]


class LLMError(RuntimeError):
    """Raised when a provider call fails, with a human-actionable message."""


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0  # estimated; the budget governor (M6) refines this


@dataclass(slots=True)
class Response:
    """Uniform result of a model turn: some text, zero or more tool calls, usage."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Every provider implements exactly one method."""

    @abstractmethod
    async def chat(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Response: ...

    async def aclose(self) -> None:  # optional cleanup hook
        return None


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying a transient failure.

    Honor a server ``Retry-After`` header when present; otherwise back off
    exponentially (0.5, 1, 2, 4 … seconds, capped at 8) so a busy endpoint gets room
    to recover without making the user wait forever.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(15.0, float(header))
        except ValueError:
            pass  # Retry-After may be an HTTP-date; fall through to plain backoff.
    return min(8.0, 0.5 * 2**attempt)


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint via httpx."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        if not base_url:
            raise LLMError(
                "No base URL configured. For MODEL_PROVIDER=custom you must set "
                "OPENAI_BASE_URL in .env."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Response:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"

        # Retry transient failures (rate limits, brief upstream hiccups) with backoff.
        # This is plain network robustness — the spend governor and kill switch are M6.
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return self._parse(resp.json())
            except httpx.ConnectError as exc:
                raise LLMError(
                    f"Could not reach {self.base_url}. If you're using Ollama, is it running "
                    f"('ollama serve') and the model pulled ('ollama pull {self.model}')?"
                ) from exc
            except httpx.TimeoutException as exc:
                raise LLMError(
                    f"Provider at {self.base_url} timed out after {self.timeout:.0f}s. The model "
                    f"may be cold-starting or under heavy load - try again, or raise "
                    f"REQUEST_TIMEOUT in .env."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRYABLE_STATUS and attempt < self.max_retries:
                    await asyncio.sleep(_retry_after(exc.response, attempt))
                    continue
                body = exc.response.text[:500]
                raise LLMError(f"Provider returned {status}: {body}") from exc

        # Unreachable: the loop either returns a Response or raises. Here for type-checkers.
        raise LLMError("Provider call failed after exhausting retries.")

    @staticmethod
    def _parse(data: dict[str, Any]) -> Response:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        text = message.get("content")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return Response(text=text, tool_calls=tool_calls, usage=usage, raw=data)

    async def aclose(self) -> None:
        await self._client.aclose()


class MockProvider(LLMProvider):
    """Deterministic provider for tests and offline demos.

    Pass a list of scripted :class:`Response` objects to simulate a multi-step run
    (e.g. "first ask for a tool, then give a final answer"). With no script, it
    simply echoes the latest user message — enough to prove the loop end to end
    with zero network and zero keys.
    """

    def __init__(self, scripted: list[Response] | None = None):
        self._scripted = list(scripted or [])

    async def chat(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Response:
        if self._scripted:
            return self._scripted.pop(0)
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return Response(text=f"(mock) {last_user}", usage=Usage())


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the provider selected in config. The single switch point."""
    if settings.model_provider == "mock":
        return MockProvider()
    return OpenAICompatibleProvider(
        base_url=settings.resolved_base_url() or "",
        model=settings.resolved_model(),
        api_key=settings.resolved_api_key(),
        timeout=settings.request_timeout,
    )


# Matches a fenced ```action { ... } ``` block (or a bare JSON object) that a model
# without native tool-calling can emit instead. Tolerant of an optional language tag.
_ACTION_BLOCK = re.compile(r"```(?:action|json|tool)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_tool_calls(text: str | None) -> list[ToolCall]:
    """Fallback parser: recover tool calls from plain text.

    Accepts either ``{"tool": "name", "args": {...}}`` or the OpenAI-ish
    ``{"name": "name", "arguments": {...}}``. Used only when the provider returned
    no native ``tool_calls``.
    """
    if not text:
        return []

    candidates = _ACTION_BLOCK.findall(text)
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [stripped]

    calls: list[ToolCall] = []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("tool") or obj.get("name")
        if not name:
            continue
        args = obj.get("args") or obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append(ToolCall(name=name, arguments=args))
    return calls
