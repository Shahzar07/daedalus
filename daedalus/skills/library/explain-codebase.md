---
name: explain-codebase
description: Map a directory or project and explain its architecture and entry points
triggers: [explain, understand, what does this project, architecture, how does this work, walk me through]
tags: [code, onboarding]
---
## When to use
The user wants to understand an unfamiliar codebase or project layout.

## Steps
1. List the directory (`shell_exec` with `ls`/`dir`, or read a manifest like
   `pyproject.toml`/`package.json`) to learn the shape and dependencies.
2. Identify the entry point(s) and the main modules.
3. `files_read` the key files to trace the primary flow of control.
4. Explain: what the project does, how it's organized, where execution starts, and how
   the major pieces talk to each other.
5. Point the user at the 2–3 files they should read first.

## Notes
- Favor a clear mental model over exhaustive detail. Name files and functions precisely.
