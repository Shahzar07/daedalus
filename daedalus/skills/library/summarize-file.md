---
name: summarize-file
description: Read a file from the workspace and produce a structured summary
triggers: [summarize, summary, tldr, what does this file, read and explain, digest]
tags: [files, writing]
---
## When to use
The user points at a file (notes, an article, code, a report) and wants the gist
without reading the whole thing.

## Steps
1. Call `files_read` on the path the user named.
2. Identify the document type (prose, data, code, config) — it changes what matters.
3. Produce: a one-line TL;DR, then 3–6 bullet points of the key content.
4. End with any action items, open questions, or notable risks you spotted.

## Notes
- Keep it faithful: summarize what's there, never add claims the file doesn't make.
- For very long files, summarize section by section, then give an overall synthesis.
