---
name: extract-data
description: Pull structured data out of messy text or a file into JSON or CSV
triggers: [extract, parse, structure, into json, into csv, pull out, scrape from text]
tags: [data, files]
---
## When to use
The user has unstructured or semi-structured content and wants clean structured output.

## Steps
1. Read the source (`files_read`) or use the text the user pasted.
2. Agree on the schema: which fields, and their types. Infer it if obvious.
3. Extract records carefully; leave a field empty/null rather than guessing a value.
4. Emit valid JSON (or CSV) — well-formed and parseable.
5. If asked, `files_write` the result to a file and report the path and record count.

## Notes
- Preserve the source's meaning; flag rows you were unsure about instead of inventing data.
