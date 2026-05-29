---
name: code-review
description: Review a source file for bugs, clarity, and style and report findings
triggers: [review, code review, look over, critique, feedback on, improve this code]
tags: [code, review]
---
## When to use
The user asks for feedback on code or wants a file checked before shipping.

## Steps
1. `files_read` the file (or files) under review.
2. Assess in this order: correctness (bugs, edge cases), then security, then clarity,
   then style/consistency.
3. Report findings grouped by severity (blocking / should-fix / nit). Quote the line.
4. For each issue, give the fix or a concrete suggestion — not just a complaint.
5. End with one or two things the code does well.

## Notes
- Be specific and kind. Prioritize the few issues that matter over a long nitpick list.
