---
name: web-research
description: Research a topic or question using web search and synthesize a sourced answer
triggers: [research, look up, find out, investigate, what is, latest, news, compare]
tags: [research, web]
---
## When to use
The user asks about something current, factual, or outside your training — anything
that benefits from looking it up rather than answering from memory.

## Steps
1. Turn the request into 1–3 focused search queries. Prefer specific terms over broad ones.
2. Call `web_search` for each query. Skim the top results.
3. If results conflict or are thin, refine the query and search again (max ~3 rounds).
4. Synthesize a concise answer in your own words. Do not paste raw results.
5. Cite the sources you used (title or URL) so the user can verify.

## Notes
- State clearly when sources disagree or when information may be out of date.
- If nothing useful comes back, say so rather than guessing.
