"""Skills — Daedalus's procedural memory.

A *skill* is a Markdown playbook (``SKILL.md``) with YAML frontmatter: a name, a
one-line description of when it applies, and a body of step-by-step guidance the
agent follows. Skills live in ``~/.dae/skills/<name>/SKILL.md`` so you can read,
edit, add, or delete them by hand.

Two halves:
  * :mod:`daedalus.skills.engine` — load skills, *match* the relevant ones to a
    request by description, and inject them into the agent's context.
  * :mod:`daedalus.skills.author` — after a successful multi-step task, ask the
    model whether the procedure is worth keeping and, if so, write a new skill.

This is how Daedalus gets better with use: solve something once the hard way, and
it can save the recipe so next time is one step.
"""
