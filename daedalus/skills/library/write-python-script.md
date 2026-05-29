---
name: write-python-script
description: Write a small Python script to the workspace and verify it runs
triggers: [write a script, python script, automate, generate code, write code, script that]
tags: [code, python, files]
---
## When to use
The user wants a self-contained Python script or utility created and working.

## Steps
1. Clarify the input/output contract only if it's genuinely ambiguous; otherwise proceed.
2. Write clean, commented code. Prefer the standard library; avoid heavy deps.
3. Save it with `files_write` to a sensibly named `.py` file in the workspace.
4. Run it with `shell_exec` (e.g. `python name.py`) to confirm it executes.
5. If it errors, read the traceback, fix the file, and re-run until it works.
6. Report the path, what it does, and how to run it.

## Notes
- Keep scripts small and single-purpose. Validate inputs and fail with clear messages.
