"""Reflection — a second pass that critiques (and maybe improves) the answer.

A single forward pass can be confidently wrong. The classic, cheap mitigation is to let
the model **review its own work**: given the original request and the drafted answer, is
it correct, complete, and safe? If yes, ship it unchanged. If not, the critic returns a
better answer, and the loop swaps it in.

This costs an extra model call per turn, so it's **off by default** (``REFLECTION=true``
to enable) and capped at ``REFLECTION_MAX_REVISIONS``. It's strictly best-effort: any
error, or an unparseable verdict, leaves the original answer untouched — reflection can
only *improve* or *no-op*, never break a turn.

The protocol is kept dead simple so parsing can't go wrong: the critic replies with the
single token ``OK`` to approve, or with the full replacement answer to revise.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.llm import LLMProvider, Usage

_CRITIC_SYS = (
    "You are a meticulous reviewer of an AI assistant's draft answer. Judge it against "
    "the user's request for correctness, completeness, and safety. If the draft is "
    "already good, reply with exactly the token OK and nothing else. Otherwise reply "
    "with an improved final answer only — no preamble, no explanation of changes."
)


@dataclass(slots=True)
class Critique:
    """Outcome of one review: the (possibly revised) answer, whether it changed, usage."""

    answer: str
    revised: bool
    usage: Usage


class Critic:
    """Reviews an answer and returns an approved or improved version."""

    def __init__(self, provider: LLMProvider, max_revisions: int = 1):
        self.provider = provider
        self.max_revisions = max(0, max_revisions)

    async def review(self, user_input: str, answer: str) -> Critique:
        """Run up to ``max_revisions`` review rounds; return the final answer + usage.

        Each round either approves the current answer (``OK`` → stop) or replaces it.
        Usage is accumulated across rounds so the budget governor can book the spend.
        """
        total = Usage()
        current = answer
        revised = False

        for _ in range(self.max_revisions):
            prompt = (
                f"User request:\n{user_input}\n\n"
                f"Draft answer:\n{current}\n\n"
                "Reply 'OK' if the draft is good, otherwise reply with the improved answer."
            )
            try:
                response = await self.provider.chat(
                    [
                        {"role": "system", "content": _CRITIC_SYS},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception:  # noqa: BLE001 - reflection is best-effort, never fatal
                break

            total = _add_usage(total, response.usage)
            verdict = (response.text or "").strip()
            # Approved, or the model returned nothing useful → keep what we have.
            if not verdict or verdict.upper() == "OK" or verdict.upper().startswith("OK"):
                break
            # A genuine revision. Adopt it and (if budget allows) review again.
            current = verdict
            revised = True

        return Critique(answer=current, revised=revised, usage=total)


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )
