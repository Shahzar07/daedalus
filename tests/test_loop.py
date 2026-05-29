"""The ReAct loop: final answers, tool execution, and the step-limit guard."""

from daedalus.config import Settings
from daedalus.core.events import EventBus, EventType
from daedalus.core.llm import MockProvider, Response, ToolCall
from daedalus.core.loop import Agent


class FakeTools:
    """Minimal ToolExecutor: one `echo` tool that records its calls."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo text back",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]

    async def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        return f"echoed:{call.arguments.get('text')}"


async def test_final_answer_without_tools():
    agent = Agent(MockProvider(), Settings(model_provider="mock"))
    out = await agent.run("hello")
    assert out == "(mock) hello"
    assert agent.history[-1] == {"role": "assistant", "content": "(mock) hello"}


async def test_executes_tool_then_answers():
    scripted = [
        Response(tool_calls=[ToolCall(name="echo", arguments={"text": "hi"}, id="call_0")]),
        Response(text="done"),
    ]
    tools = FakeTools()
    bus = EventBus()
    seen: list[EventType] = []
    bus.subscribe(lambda e: seen.append(e.type))

    agent = Agent(MockProvider(scripted), Settings(model_provider="mock"), bus=bus, tools=tools)
    out = await agent.run("please echo hi")

    assert out == "done"
    assert len(tools.calls) == 1 and tools.calls[0].name == "echo"
    assert {EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.FINAL} <= set(seen)


async def test_step_limit_guard_stops_runaway():
    # The model keeps asking for a tool and never finalizes.
    scripted = [
        Response(tool_calls=[ToolCall(name="echo", arguments={"text": "x"}, id="c")])
        for _ in range(10)
    ]
    agent = Agent(
        MockProvider(scripted),
        Settings(model_provider="mock", max_steps=3),
        tools=FakeTools(),
    )
    out = await agent.run("loop forever")
    assert "step limit" in out


async def test_json_fallback_tool_path():
    # No native tool_calls — the model emits an ```action`` block instead.
    scripted = [
        Response(text='```action\n{"tool": "echo", "args": {"text": "hey"}}\n```'),
        Response(text="all done"),
    ]
    tools = FakeTools()
    agent = Agent(MockProvider(scripted), Settings(model_provider="mock"), tools=tools)
    out = await agent.run("echo hey via fallback")
    assert out == "all done"
    assert tools.calls and tools.calls[0].arguments == {"text": "hey"}
