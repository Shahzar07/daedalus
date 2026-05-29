---
name: debug-error
description: Systematically diagnose and fix an error message or failing program
triggers: [error, exception, traceback, not working, fails, bug, broken, debug, fix]
tags: [code, debugging]
---
## When to use
The user shares an error, stack trace, or describes something that isn't working.

## Steps
1. Read the error carefully: the exception type, the message, and the deepest frame
   in the user's own code (not library internals).
2. Form a single most-likely hypothesis for the root cause.
3. Gather evidence: `files_read` the implicated file/lines; if useful, reproduce with
   `shell_exec`.
4. Apply the smallest fix that addresses the root cause (use `files_patch`).
5. Re-run to confirm the error is gone and nothing else broke.
6. Explain what was wrong and why the fix works — briefly.

## Notes
- Change one thing at a time. If the first hypothesis is wrong, revise it, don't pile on.
