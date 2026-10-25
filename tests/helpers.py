"""Small helpers shared by the tests."""

from agent_core.llm import LLMResponse, ToolCall


class FakeLLM:
    """Replays scripted replies instead of calling a real model."""

    provider = "fake"
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": [t["name"] for t in tools or []]})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_reply(*calls):
    """Build an LLM reply that asks for the given (tool_name, arguments) calls."""
    return LLMResponse("", [ToolCall(f"call{i}", name, args) for i, (name, args) in enumerate(calls)])
