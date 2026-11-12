"""Regression tests for async-generator tool consumption in AsyncToolDispatcher.

dispatch() must collect async-generator results and return a serialized str;
only synchronous generator tools are passed through for chunked streaming.
"""
import asyncio
import json
from types import SimpleNamespace

from LightAgent import AsyncToolDispatcher, LightAgent


def make_agent(**kwargs):
    return LightAgent(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        auto_discover_skills=False,
        **kwargs,
    )


def attach_client(agent, completions):
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return completions


def tool_messages_of(params):
    return [m for m in params.get("messages", []) if isinstance(m, dict) and m.get("role") == "tool"]


side_effects = []


def reset_effects():
    side_effects.clear()


async def async_gen_tool():
    side_effects.append("entered")
    for i in range(3):
        yield f"chunk-{i}"


async_gen_tool.tool_info = {
    "tool_name": "async_gen_tool",
    "tool_description": "async generator tool",
    "tool_params": [],
}


async def async_gen_single_dict_tool():
    side_effects.append("entered")
    yield {"answer": 42}


async_gen_single_dict_tool.tool_info = {
    "tool_name": "async_gen_single_dict_tool",
    "tool_description": "async generator yielding one dict",
    "tool_params": [],
}


async def async_gen_empty_tool():
    side_effects.append("entered")
    return
    yield  # unreachable; makes this an async generator function


async_gen_empty_tool.tool_info = {
    "tool_name": "async_gen_empty_tool",
    "tool_description": "async generator yielding nothing",
    "tool_params": [],
}


def sync_gen_tool():
    side_effects.append("entered")
    for i in range(3):
        yield f"chunk-{i}"


sync_gen_tool.tool_info = {
    "tool_name": "sync_gen_tool",
    "tool_description": "sync generator tool",
    "tool_params": [],
}


async def async_plain_tool():
    side_effects.append("ran")
    return "async-plain-result"


async_plain_tool.tool_info = {
    "tool_name": "async_plain_tool",
    "tool_description": "plain async function tool",
    "tool_params": [],
}


class TwoStepCompletions:
    """First create() returns a tool call, second returns a plain reply."""

    def __init__(self, tool_name, tool_args=None):
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name=self.tool_name,
                    arguments=json.dumps(self.tool_args),
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        message = SimpleNamespace(content="final-reply", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _delta_chunk(content=None, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(reasoning_content=None, content=content, tool_calls=None),
        finish_reason=finish_reason,
    )])


class TwoStepStreamCompletions:
    """First create() returns tool-call deltas, second returns content deltas."""

    def __init__(self, tool_name, tool_args=None):
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                index=0, id="call_1",
                function=SimpleNamespace(
                    name=self.tool_name,
                    arguments=json.dumps(self.tool_args),
                ),
            )
            delta = SimpleNamespace(reasoning_content=None, content=None, tool_calls=[tool_call])
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)]),
                _delta_chunk(finish_reason="tool_calls"),
            ])
        return iter([_delta_chunk("final-reply"), _delta_chunk(finish_reason="stop")])


def test_dispatch_collects_asyncgen_tool():
    reset_effects()
    dispatcher = AsyncToolDispatcher(
        {"async_gen_tool": async_gen_tool},
        {"async_gen_tool": async_gen_tool.tool_info},
    )

    result = asyncio.run(dispatcher.dispatch("async_gen_tool", {}))

    assert result == "chunk-0chunk-1chunk-2"
    assert side_effects == ["entered"]


def test_dispatch_serializes_single_dict_chunk():
    reset_effects()
    dispatcher = AsyncToolDispatcher(
        {"async_gen_single_dict_tool": async_gen_single_dict_tool},
        {"async_gen_single_dict_tool": async_gen_single_dict_tool.tool_info},
    )

    result = asyncio.run(dispatcher.dispatch("async_gen_single_dict_tool", {}))

    assert result == json.dumps({"answer": 42}, ensure_ascii=False)
    assert side_effects == ["entered"]


def test_dispatch_empty_asyncgen_returns_empty_string():
    reset_effects()
    dispatcher = AsyncToolDispatcher(
        {"async_gen_empty_tool": async_gen_empty_tool},
        {"async_gen_empty_tool": async_gen_empty_tool.tool_info},
    )

    result = asyncio.run(dispatcher.dispatch("async_gen_empty_tool", {}))

    assert result == ""
    assert side_effects == ["entered"]


def test_nonstream_run_consumes_asyncgen_tool():
    reset_effects()
    agent = make_agent()
    completions = attach_client(agent, TwoStepCompletions("async_gen_tool"))

    agent.run("use the tool", tools=[async_gen_tool])

    assert side_effects == ["entered"]
    tool_msgs = tool_messages_of(completions.calls[1])
    assert tool_msgs and tool_msgs[-1]["content"] == "chunk-0chunk-1chunk-2"


def test_stream_run_consumes_asyncgen_tool_without_placeholder():
    reset_effects()
    agent = make_agent()
    completions = attach_client(agent, TwoStepStreamCompletions("async_gen_tool"))

    outputs = []
    for event in agent.run("use the tool", tools=[async_gen_tool], stream=True,
                           max_retry=2, max_tool_iterations=3):
        if isinstance(event, dict) and "output" in event and event.get("name") == "async_gen_tool":
            outputs.append(event["output"])

    joined = "".join(outputs)
    assert joined == "chunk-0chunk-1chunk-2"
    assert side_effects == ["entered"]
    assert "<async_generator" not in joined
    tool_msgs = tool_messages_of(completions.calls[1])
    assert tool_msgs and tool_msgs[-1]["content"] == "chunk-0chunk-1chunk-2"
    assert "<async_generator" not in tool_msgs[-1]["content"]


def test_nonstream_syncgen_tool_still_consumed():
    reset_effects()
    agent = make_agent()
    completions = attach_client(agent, TwoStepCompletions("sync_gen_tool"))

    agent.run("use the tool", tools=[sync_gen_tool])

    assert side_effects == ["entered"]
    tool_msgs = tool_messages_of(completions.calls[1])
    assert tool_msgs and tool_msgs[-1]["content"] == "chunk-0chunk-1chunk-2"


def test_stream_syncgen_tool_still_streams_chunks():
    reset_effects()
    agent = make_agent()
    attach_client(agent, TwoStepStreamCompletions("sync_gen_tool"))

    outputs = []
    for event in agent.run("use the tool", tools=[sync_gen_tool], stream=True,
                           max_retry=2, max_tool_iterations=3):
        if isinstance(event, dict) and "output" in event and event.get("name") == "sync_gen_tool":
            outputs.append(event["output"])

    assert "".join(outputs) == "chunk-0chunk-1chunk-2"
    assert side_effects == ["entered"]


def test_nonstream_plain_async_tool_still_consumed():
    reset_effects()
    agent = make_agent()
    completions = attach_client(agent, TwoStepCompletions("async_plain_tool"))

    agent.run("use the tool", tools=[async_plain_tool])

    assert side_effects == ["ran"]
    tool_msgs = tool_messages_of(completions.calls[1])
    assert tool_msgs and tool_msgs[-1]["content"] == "async-plain-result"
