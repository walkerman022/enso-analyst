import asyncio
import json
from functools import wraps
from types import SimpleNamespace

import pytest

from LightAgent import (
    BaseCapabilityProvider,
    AsyncToolDispatcher,
    CapabilitySpec,
    LightAgent,
    SafeExpressionError,
    evaluate_safe_expression,
)
from LightAgent.builtin_tools.python_executor import (
    UNSAFE_PYTHON_TOOL_NAMES,
    execute_python_code,
)
from LightAgent.capabilities import ToolProviderAdapter


def make_agent(**kwargs):
    return LightAgent(model="security-test-model", api_key="test-key", **kwargs)


def test_safe_expression_evaluates_data_only_operations():
    assert evaluate_safe_expression("45 * 9827") == 442215
    assert evaluate_safe_expression("[1, 2, 3]") == [1, 2, 3]
    assert evaluate_safe_expression("3 < 4 and 8 != 9") is True


@pytest.mark.parametrize("expression", [
    "__import__('os')",
    "open('secret.txt')",
    "(1).__class__",
    "[1][0]",
    "{[1]: 2}",
    "{x for x in range(3)}",
    "lambda: 1",
])
def test_safe_expression_rejects_execution_and_runtime_access(expression):
    with pytest.raises(SafeExpressionError):
        evaluate_safe_expression(expression)


def test_safe_expression_rejects_resource_exhaustion_inputs():
    expression = "[0] * 65"
    with pytest.raises(SafeExpressionError):
        evaluate_safe_expression(expression)


def test_arbitrary_python_tools_are_not_registered_by_default():
    agent = make_agent()
    names = set(agent.tool_registry.function_mappings)

    assert "safe_expression" in names
    assert "execute_python_code" not in names
    assert "execute_python_file" not in names
    assert "execute_python_code_stream" not in names


def test_explicit_arbitrary_python_opt_in_still_requires_sandbox():
    agent = make_agent(enable_unsafe_python=True)
    agent.runtime.open_session()

    assert "execute_python_code" in agent.tool_registry.function_mappings
    _, error = agent._prepare_tool_call("execute_python_code", {"code": "1 + 1"})

    assert error is not None
    assert "LA-SANDBOX" in error
    assert "SandboxProvider" in error


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("tool_name", sorted(UNSAFE_PYTHON_TOOL_NAMES))
def test_sandbox_registration_does_not_open_legacy_executor_gate(started, tool_name):
    class TestSandboxProvider(BaseCapabilityProvider):
        name = "sandbox"

        def __init__(self):
            super().__init__([CapabilitySpec("sandbox.execute", execute=True)])

    agent = make_agent(
        enable_unsafe_python=True,
        capability_registry=None,
    )
    agent.runtime.open_session()
    provider = TestSandboxProvider()
    agent.capability_registry.register(provider)
    if started:
        asyncio.run(provider.mount(agent.runtime.context))
        asyncio.run(provider.start())

    _, error = agent._prepare_tool_call(tool_name, {"code": "1 + 1"})

    assert "LA-SANDBOX" in error


@pytest.mark.parametrize("tool_name", sorted(UNSAFE_PYTHON_TOOL_NAMES))
def test_direct_dispatch_and_tool_provider_block_legacy_names(tool_name):
    calls = []

    def unexpected_execution(**kwargs):
        calls.append(kwargs)
        raise AssertionError("legacy executor must not be called")

    agent = make_agent(enable_unsafe_python=True)
    agent.tool_registry.function_mappings[tool_name] = unexpected_execution
    dispatcher = AsyncToolDispatcher(agent.tool_registry.function_mappings)
    assert "LA-SANDBOX" in asyncio.run(dispatcher.dispatch(tool_name, {}))
    provider = ToolProviderAdapter(agent.tool_registry)
    assert "LA-SANDBOX" in asyncio.run(provider.invoke(f"tool.{tool_name}"))
    assert calls == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_builtin_executor_alias_is_blocked(wrapped):
    @wraps(execute_python_code)
    def wrapper(**kwargs):
        raise AssertionError("wrapped executor must not be called")

    tool = wrapper if wrapped else execute_python_code
    agent = make_agent()
    agent.runtime.open_session()
    agent.tool_registry.function_mappings["legacy_alias"] = tool
    _, error = agent._prepare_tool_call("legacy_alias", {"code": "1 + 1"})
    assert "LA-SANDBOX" in error
    dispatcher = AsyncToolDispatcher({"legacy_alias": tool})
    assert "LA-SANDBOX" in asyncio.run(dispatcher.dispatch("legacy_alias", {}))


class LegacyToolCompletions:
    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.tool_results = []

    def create(self, **params):
        tool_results = [
            m["content"] for m in params["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        if tool_results:
            self.tool_results = tool_results
            if params.get("stream"):
                delta = SimpleNamespace(content="blocked", tool_calls=None, reasoning_content=None)
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")])])
            message = SimpleNamespace(content="blocked", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        function = SimpleNamespace(name=self.tool_name, arguments=json.dumps({"code": "1 + 1"}))
        call = SimpleNamespace(id="legacy-call", index=0, function=function)
        if params.get("stream"):
            delta = SimpleNamespace(content=None, tool_calls=[call], reasoning_content=None)
            end = SimpleNamespace(content=None, tool_calls=None, reasoning_content=None)
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)]),
                SimpleNamespace(choices=[SimpleNamespace(delta=end, finish_reason="tool_calls")]),
            ])
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_name", sorted(UNSAFE_PYTHON_TOOL_NAMES))
def test_model_cannot_execute_legacy_tool_in_either_run_mode(stream, tool_name, monkeypatch):
    agent = make_agent(enable_unsafe_python=True)
    calls = []

    def unexpected_execution(**kwargs):
        calls.append(kwargs)
        raise AssertionError("legacy executor must not be called")

    monkeypatch.setitem(agent.tool_registry.function_mappings, tool_name, unexpected_execution)
    completions = LegacyToolCompletions(tool_name)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = agent.run("calculate", stream=stream, trace=True, max_retry=2)
    if stream:
        list(result)
    assert completions.tool_results
    assert all("LA-SANDBOX" in value for value in completions.tool_results)
    assert calls == []
    assert any(event["type"] == "error" for event in agent.export_trace())
