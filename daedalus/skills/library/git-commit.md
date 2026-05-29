---
name: git-commit
description: Stage changes and write a clear, conventional git commit message
triggers: [commit, git commit, save changes, check in, stage]
tags: [git, workflow]
---
## When to use
The user wants to commit work in a git repository.

## Steps
1. Run `git status` and `git diff` (via `shell_exec`) to see exactly what changed.
2. Group the changes into one logical commit (or suggest splitting if they're unrelated).
3. Write a message: a concise imperative subject (<=50 chars), a blank line, then a
   short body explaining the *why* when it isn't obvious.
4. Stage the relevant files by name (avoid blanket `git add -A` if there are stray files).
5. Create the commit and show `git status` to confirm.

## Notes
- Never commit secrets or large binaries. Flag them instead of committing.
- Don't push unless the user explicitly asks.
