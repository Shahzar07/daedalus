---
name: daily-briefing
description: Compile a short briefing on one or more topics from fresh web searches
triggers: [briefing, daily, digest, roundup, catch me up, what happened, update on]
tags: [research, web, scheduler]
---
## When to use
The user wants a compact status on topics they care about — great for a scheduled,
recurring job (see `dae jobs`).

## Steps
1. Take the topic list from the request (or from prior context/memory).
2. `web_search` each topic for the most recent, relevant items.
3. For each topic, write 2–4 tight bullets: what's new and why it matters.
4. Lead with the single most important development across all topics.
5. Keep the whole briefing skimmable in under a minute.

## Notes
- Skip topics with nothing new rather than padding. Cite sources for anything surprising.
