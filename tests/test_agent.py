from agent import ToolCallingAgent
from openrouter import ChatResponse
from tools import ToolRegistry


class FakeToolClient:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        if self.calls == 1:
            return ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_add",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                    }
                ],
                raw={},
            )
        return ChatResponse(content="The answer is 5.", tool_calls=[], raw={})


def test_tool_calling_agent_runs_multi_step_tool_loop() -> None:
    registry = ToolRegistry()
    registry.register(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        handler=lambda a, b: a + b,
    )
    client = FakeToolClient()

    result = ToolCallingAgent(client=client, tools=registry, max_steps=3).run("What is 2 + 3?")

    assert result.final_content == "The answer is 5."
    assert len(result.tool_results) == 1
    assert client.calls == 2
    assert any(message["role"] == "tool" for message in client.messages_seen[-1])
