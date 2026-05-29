"""Web search tool — free and keyless by default.

Uses a self-hosted SearXNG instance if ``SEARXNG_URL`` is set, otherwise falls back
to DuckDuckGo via the ``ddgs`` library (no API key, no account). Both are imported
lazily so a missing/optional backend never breaks tool discovery.
"""

from __future__ import annotations

from ..config import get_settings
from .registry import Tool


def _search_searxng(query: str, max_results: int, base_url: str) -> list[tuple[str, str, str]]:
    import httpx

    resp = httpx.get(
        f"{base_url.rstrip('/')}/search",
        params={"q": query, "format": "json"},
        timeout=20.0,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:max_results]
    return [(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in results]


def _search_duckduckgo(query: str, max_results: int) -> list[tuple[str, str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:  # older releases shipped as duckduckgo_search
        from duckduckgo_search import DDGS

    rows: list[tuple[str, str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            rows.append((r.get("title", ""), r.get("href", ""), r.get("body", "")))
    return rows


def web_search(query: str, max_results: int = 5) -> str:
    settings = get_settings()
    try:
        if settings.searxng_url:
            rows = _search_searxng(query, max_results, settings.searxng_url)
        else:
            rows = _search_duckduckgo(query, max_results)
    except Exception as exc:  # noqa: BLE001 - report failures to the model, not a crash
        return f"ERROR: web search failed: {exc}"

    if not rows:
        return f"No results for: {query}"

    lines = [f"Results for: {query}"]
    for i, (title, url, snippet) in enumerate(rows, 1):
        snippet = " ".join((snippet or "").split())
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)


TOOL = Tool(
    name="web_search",
    description="Search the web and return the top results (title, URL, snippet). "
    "Uses DuckDuckGo for free, or a configured SearXNG instance.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "how many results (default 5)"},
        },
        "required": ["query"],
    },
    func=web_search,
)
