"""THE agent loop — the one file to read if you read nothing else.

This is a hand-rolled **ReAct** loop (Reason + Act). There is no framework hiding
the control flow: you can see every step the agent takes. One call to
:meth:`Agent.run` is one *request-scoped* loop — it begins when the user sends a
message and ends when the agent produces a final answer (or hits the step limit).
It never runs on its own.

Each iteration:

  1. Assemble the messages the model will see (system prompt + history + input).
  2. Call the model, optionally offering it tools.
  3. Parse the reply into either a *final answer* or one/more *tool calls*
     (native function-calls first; a plain-text ```action`` block as a fallback).
  4. If there are tool calls: run them, append their results, and loop again.
  5. If there are none: that's the final answer. Stop.

Every step publishes an :class:`Event`, so surfaces and the audit log can watch the
agent think without the loop knowing or caring who's listening.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..memory.files import read_memory_md, read_soul_md, read_user_md
from ..memory.store import MemoryStore
from ..safety.budget import BudgetError, BudgetGovernor
from ..safety.guardrails import Guardrails
from ..skills.author import maybe_author_skill
from ..skills.engine import Skill, SkillLibrary
from .context import assemble_messages
from .events import Event, EventBus, EventType, get_event_bus
from .llm import LLMProvider, Message, Response, ToolCall, ToolSchema, extract_json_tool_calls

if TYPE_CHECKING:
    from ..memory.graph import MemoryGraph
    from ..memory.semantic import SemanticIndex
    from ..reflection.critic import Critic
    from ..subagents.spawn import SubagentSpawner

# A focused prompt for the post-turn "what's worth remembering?" step.
_SUMMARIZE_SYS = (
    "You decide what, if anything, from a conversation turn is worth remembering "
    "long-term about the user or their world. Be conservative — most turns have "
    "nothing worth keeping. Never invent facts."
)


@runtime_checkable
class ToolExecutor(Protocol):
    """What the loop needs from the tool system (implemented by the registry in M2).

    Kept as a Protocol so the loop depends on a tiny interface, not the whole tool
    subsystem. In M1 there is no executor and the agent simply answers from the model.
    """

    def schemas(self) -> list[ToolSchema]:
        """OpenAI-format tool schemas to advertise to the model."""
        ...

    async def execute(self, call: ToolCall) -> str:
        """Run one tool call and return its result as a string."""
        ...


class Agent:
    """Holds the conversation and runs the loop.

    Conversation ``history`` persists *between* requests so follow-up questions have
    context — that's ordinary chat memory, not the loop running unattended.
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings | None = None,
        bus: EventBus | None = None,
        tools: ToolExecutor | None = None,
        memory: MemoryStore | None = None,
        skills: SkillLibrary | None = None,
        budget: BudgetGovernor | None = None,
        guardrails: Guardrails | None = None,
        graph: MemoryGraph | None = None,
        semantic: SemanticIndex | None = None,
        critic: Critic | None = None,
        subagents: SubagentSpawner | None = None,
    ):
        self.provider = provider
        self.settings = settings or get_settings()
        self.bus = bus or get_event_bus()
        self.tools = tools
        self.memory = memory
        self.skills = skills
        # Safety rails. Both optional so the loop still runs in minimal builds and
        # tests; when present they gate spend (budget + kill switch) and authorize
        # world-changing tool calls (guardrails). See :mod:`daedalus.safety`.
        self.budget = budget
        self.guardrails = guardrails
        # M10 upgrades, all optional and individually wired in _build_agent:
        #   graph     - entity/relation recall alongside FTS (memory/graph.py)
        #   semantic  - recall-by-meaning via embeddings (memory/semantic.py)
        #   critic    - post-answer self-review pass (reflection/critic.py)
        #   subagents - lets the agent delegate bounded subtasks (subagents/spawn.py)
        self.graph = graph
        self.semantic = semantic
        self.critic = critic
        self.subagents = subagents
        self.history: list[Message] = []

    async def run(self, user_input: str, run_id: str | None = None) -> str:
        """Execute one request-scoped ReAct loop and return the final answer."""
        run_id = run_id or uuid.uuid4().hex[:8]

        # Reset the per-request subagent budget so the "max children per request" cap is
        # measured from here, not across the whole session.
        if self.subagents is not None:
            self.subagents.begin_run()

        # Advertise tools (if any) to the model. Empty in M1.
        tool_schemas = self.tools.schemas() if self.tools else None
        tool_names = [s["function"]["name"] for s in tool_schemas] if tool_schemas else None

        # Gather memory to put in front of the model: the always-on Markdown files
        # (persona + world facts + user profile) plus any stored facts that look
        # relevant to *this* input (FTS5 recall).
        soul_md, memory_md, user_md, recalled = self._gather_memory(user_input)

        # Match saved skills (playbooks) to this request and inject the best few.
        matched_skills = self._gather_skills(user_input)
        if matched_skills:
            await self.bus.emit(
                Event(
                    EventType.DECISION,
                    run_id,
                    {"skills_matched": [s.name for s in matched_skills]},
                    0,
                )
            )

        # Working transcript for this request: system prompt + prior turns + new input.
        messages = assemble_messages(
            self.history,
            user_input,
            tool_names,
            memory_md,
            user_md,
            recalled,
            matched_skills,
            soul_md,
        )

        tools_used: list[str] = []
        final_text = ""
        halted = False
        for step in range(1, self.settings.max_steps + 1):
            # --- 0: safety gate — consult the kill switch + daily cap BEFORE we
            # spend anything. A halt here is a graceful stop, not a crash: we tell
            # the user why and break, leaving the transcript intact. -------------
            if self.budget is not None:
                try:
                    self.budget.check()
                except BudgetError as exc:
                    await self.bus.emit(
                        Event(EventType.DECISION, run_id, {"halted": str(exc)}, step)
                    )
                    final_text = f"(stopped: {exc})"
                    halted = True
                    break

            # --- 1 & 2: ask the model ---------------------------------------
            response = await self.provider.chat(messages, tools=tool_schemas)
            await self._emit_usage(run_id, step, response)
            # Book the spend right away so the next iteration's check() sees it and
            # the /budget command reflects this turn.
            if self.budget is not None:
                self.budget.record(response.usage)

            # --- 3: parse — native tool calls first, JSON fallback second ----
            native = bool(response.tool_calls)
            tool_calls = response.tool_calls or extract_json_tool_calls(response.text)

            if response.text:
                await self.bus.emit(Event(EventType.THOUGHT, run_id, {"text": response.text}, step))

            # --- 5: no tool calls => this is the final answer ----------------
            if not tool_calls:
                final_text = response.text or ""
                messages.append({"role": "assistant", "content": final_text})
                await self.bus.emit(Event(EventType.FINAL, run_id, {"text": final_text}, step))
                break

            # --- 4: run the requested tools and feed results back ------------
            tools_used.extend(call.name for call in tool_calls)
            self._record_assistant_request(messages, response, tool_calls, native)
            await self._run_tools(messages, tool_calls, native, run_id, step)
        else:
            # Loop exhausted without a final answer — a guardrail, not a crash.
            await self.bus.emit(
                Event(
                    EventType.DECISION,
                    run_id,
                    {"reason": "max_steps_reached"},
                    self.settings.max_steps,
                )
            )
            if not final_text:
                final_text = "(stopped: reached the step limit without a final answer)"

        # Reflection (M10): an optional self-review pass that may improve the answer
        # before we commit it. Skipped after a halt (we've decided not to spend more).
        if not halted and self.critic is not None and final_text:
            final_text = await self._maybe_reflect(user_input, final_text, messages, run_id)

        # Commit this request's transcript (minus the system prompt) as history so the
        # next request continues the conversation.
        self.history = messages[1:]

        # Persist this turn for long-term recall, then let the model decide whether
        # anything here is worth keeping as a durable fact. Both are best-effort: a
        # memory hiccup must never sink the answer the user already has.
        if self.memory is not None:
            self.memory.archive_message("user", user_input, run_id)
            self.memory.archive_message("assistant", final_text, run_id)
            # Skip the reflection step if we halted on budget/kill switch — it would
            # only make another model call we've already decided not to afford.
            if not halted:
                await self._maybe_remember(user_input, final_text, run_id)

        # If this turn used tools to accomplish something we *didn't* already have a
        # playbook for, consider saving the recipe as a reusable skill. Skipping turns
        # that matched an existing skill avoids near-duplicates and a wasted LLM call.
        # Best-effort — Daedalus learning, never blocking. (And never after a halt.)
        if not halted and self.skills is not None and tools_used and not matched_skills:
            await self._maybe_author_skill(user_input, final_text, tools_used, run_id)

        return final_text

    # ---- helpers (kept small so run() reads top to bottom) -------------------

    def _gather_skills(self, user_input: str) -> list[Skill]:
        """Pick the saved skills most relevant to this request (empty if none)."""
        if self.skills is None:
            return []
        return self.skills.match(user_input, limit=3)

    async def _maybe_author_skill(
        self, user_input: str, final_text: str, tools_used: list[str], run_id: str
    ) -> None:
        """Post-turn: maybe distill this task into a new reusable skill."""
        if self.skills is None:
            return
        try:
            skill = await maybe_author_skill(
                self.provider, self.skills, user_input, final_text, tools_used
            )
        except Exception:  # noqa: BLE001 - authoring is best-effort, never fatal
            return
        if skill is not None:
            await self.bus.emit(
                Event(EventType.DECISION, run_id, {"skill_authored": skill.name}, 0)
            )

    def _gather_memory(self, user_input: str) -> tuple[str, str, str, list[str]]:
        """Collect the memory context for this turn.

        Returns ``(soul_md, memory_md, user_md, recalled)``. The three Markdown files are
        always-on persona/world/profile context (``SOUL.md`` is read even with no FTS
        store wired, since the persona is independent of recall). ``recalled`` unifies up
        to three sources, in priority order: keyword recall (FTS), semantic recall
        (embeddings, M10), and connected facts from the memory graph (M10). They're
        de-duplicated and capped so the prompt stays small. With no store wired up the
        recall list is empty, so the prompt is exactly what M1/M2 produced — memory is
        purely additive.
        """
        soul_md = read_soul_md()
        if self.memory is None:
            return soul_md, "", "", []

        recalled: list[str] = list(self.memory.recall(user_input))

        if self.semantic is not None:
            try:
                for hit in self.semantic.search(user_input):
                    if hit not in recalled:
                        recalled.append(hit)
            except Exception:  # noqa: BLE001 - recall is best-effort, never fatal
                pass

        if self.graph is not None:
            try:
                for edge in self.graph.related(user_input):
                    if edge not in recalled:
                        recalled.append(edge)
            except Exception:  # noqa: BLE001 - recall is best-effort, never fatal
                pass

        return soul_md, read_memory_md(), read_user_md(), recalled[:8]

    async def _maybe_remember(self, user_input: str, final_text: str, run_id: str) -> None:
        """Post-turn reflection: ask the model whether this turn taught us anything
        durable, and store it if so.

        Convention keeps the parser trivial: the model replies with one ``REMEMBER:``
        line per fact, or the single word ``NONE``. Anything we don't recognize is
        treated as "nothing to keep" — we'd rather forget than fabricate a memory.
        """
        if self.memory is None:
            return

        prompt = (
            "Conversation turn:\n"
            f"User: {user_input}\n"
            f"Assistant: {final_text}\n\n"
            "Is there a durable fact about the user or their world worth remembering? "
            "Reply with one line per fact, each beginning 'REMEMBER:', or exactly 'NONE'."
        )
        try:
            response = await self.provider.chat(
                [
                    {"role": "system", "content": _SUMMARIZE_SYS},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:  # noqa: BLE001 - reflection is best-effort, never fatal
            return

        stored: list[str] = []
        for line in (response.text or "").splitlines():
            line = line.strip()
            if line.upper().startswith("REMEMBER:"):
                fact = line[len("REMEMBER:") :].strip()
                if fact:
                    self.memory.remember(fact)
                    # Mirror the fact into the M10 indexes so future recall can find it
                    # by meaning (semantic) and by connection (graph). Best-effort.
                    self._index_fact(fact)
                    stored.append(fact)

        if stored:
            await self.bus.emit(
                Event(EventType.DECISION, run_id, {"remembered": stored}, self.settings.max_steps)
            )

    def _index_fact(self, fact: str) -> None:
        """Add a freshly remembered fact to the semantic index and memory graph."""
        if self.semantic is not None:
            try:
                self.semantic.add(fact)
            except Exception:  # noqa: BLE001 - indexing is best-effort, never fatal
                pass
        if self.graph is not None:
            try:
                self.graph.add_fact(fact)
            except Exception:  # noqa: BLE001 - indexing is best-effort, never fatal
                pass

    async def _maybe_reflect(
        self, user_input: str, final_text: str, messages: list[Message], run_id: str
    ) -> str:
        """Run the reflection critic; if it improves the answer, swap it into the
        transcript and return the revision. Spend is booked against the budget."""
        # Respect the kill switch / daily cap before spending another model call.
        if self.budget is not None:
            try:
                self.budget.check()
            except BudgetError:
                return final_text
        try:
            critique = await self.critic.review(user_input, final_text)
        except Exception:  # noqa: BLE001 - reflection can only help, never break a turn
            return final_text

        if self.budget is not None:
            self.budget.record(critique.usage)

        if critique.revised and critique.answer.strip():
            # Replace the last assistant message (the draft) with the improved answer.
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    msg["content"] = critique.answer
                    break
            await self.bus.emit(Event(EventType.DECISION, run_id, {"reflection": "revised"}, 0))
            return critique.answer

        await self.bus.emit(Event(EventType.DECISION, run_id, {"reflection": "approved"}, 0))
        return final_text

    async def _emit_usage(self, run_id: str, step: int, response: Response) -> None:
        u = response.usage
        await self.bus.emit(
            Event(
                EventType.USAGE,
                run_id,
                {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                    "cost_usd": u.cost_usd,
                },
                step,
            )
        )

    def _record_assistant_request(
        self,
        messages: list[Message],
        response: Response,
        tool_calls: list[ToolCall],
        native: bool,
    ) -> None:
        """Append the assistant turn that asked for tools, so the model sees its own
        request on the next iteration. Native calls use the OpenAI tool-call shape;
        fallback calls just keep the assistant's text (results arrive as a user turn)."""
        if native:
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for i, call in enumerate(tool_calls)
                    ],
                }
            )
        elif response.text:
            messages.append({"role": "assistant", "content": response.text})

    async def _run_tools(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        native: bool,
        run_id: str,
        step: int,
    ) -> None:
        observations: list[str] = []
        for i, call in enumerate(tool_calls):
            await self.bus.emit(
                Event(
                    EventType.TOOL_CALL, run_id, {"name": call.name, "args": call.arguments}, step
                )
            )

            # Authorization gate: a tool flagged requires_approval must clear the
            # guardrail policy (auto / ask / deny) before it runs. A blocked call is
            # not an error — we hand the model a "blocked" observation so it can choose
            # another path or explain that it needs permission.
            allowed, reason = True, ""
            if self.guardrails is not None:
                allowed, reason = await self.guardrails.authorize(call)

            if not allowed:
                result = f"(blocked by guardrails: {reason})"
                await self.bus.emit(
                    Event(
                        EventType.DECISION, run_id, {"blocked": call.name, "reason": reason}, step
                    )
                )
            elif self.tools is None:
                result = "(no tools are available in this build)"
            else:
                try:
                    result = await self.tools.execute(call)
                except Exception as exc:  # noqa: BLE001 - surface, don't crash the run
                    result = f"ERROR running {call.name}: {exc}"

            await self.bus.emit(
                Event(EventType.TOOL_RESULT, run_id, {"name": call.name, "result": result}, step)
            )

            if native:
                # OpenAI convention: a `tool` message keyed to the call id.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{i}",
                        "name": call.name,
                        "content": result,
                    }
                )
            else:
                observations.append(f"Observation from {call.name}: {result}")

        # For the text-fallback path, hand all observations back as one user turn.
        if not native and observations:
            messages.append({"role": "user", "content": "\n".join(observations)})
