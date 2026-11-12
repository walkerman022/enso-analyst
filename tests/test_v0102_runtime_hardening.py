import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

import LightAgent.flow as flow_module
from LightAgent import AgentRuntime, CancellationToken, LightAgent, LightFlow, RunResult


class SequenceAgent:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append({"query": query, "kwargs": kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_raised_agent_exception_retries_and_redacts_details():
    agent = SequenceAgent(
        "flaky",
        [RuntimeError("Authorization: Bearer secret-token"), "recovered"],
    )

    result = LightFlow().step("work", agent=agent, max_retry=2).run("go")

    assert result.success is True
    assert result.content == "recovered"
    assert result.steps[0].attempts == 2


def test_failed_agent_exception_does_not_expose_secret_details():
    agent = SequenceAgent(
        "failing",
        [RuntimeError("Authorization: Bearer secret-token")],
    )

    result = LightFlow().step("work", agent=agent).run("go")

    assert result.success is False
    assert "RuntimeError" in result.error
    assert "secret-token" not in result.error
    assert "Authorization" not in result.error


def test_raised_agent_exception_uses_fallback_and_failed_checkpoint():
    fallback = SequenceAgent("fallback", ["fallback result"])
    flow = LightFlow().step(
        "work",
        agent=SequenceAgent("primary", [RuntimeError("provider failed")]),
        fallback_agent=fallback,
    )

    result = flow.run("go", run_id="exception-fallback")

    assert result.success is True
    assert result.content == "fallback result"
    assert result.steps[0].used_fallback is True
    assert flow.get_run("exception-fallback")["status"] == "success"


def test_soft_timeout_returns_promptly_without_implicit_overlap():
    finished = Event()

    class SlowAgent:
        name = "slow"

        def run(self, query, **kwargs):
            time.sleep(0.3)
            finished.set()
            return "late"

    fallback = SequenceAgent("fallback", ["must not run"])
    flow = LightFlow().step(
        "work",
        agent=SlowAgent(),
        timeout=0.02,
        max_retry=2,
        fallback_agent=fallback,
    )

    started = time.perf_counter()
    result = flow.run("go")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
    assert result.success is False
    assert result.steps[0].attempts == 1
    assert result.steps[0].timed_out is True
    assert result.steps[0].timeout_mode == "soft"
    assert fallback.calls == []
    assert not finished.is_set()
    finished.wait(timeout=1)


def test_timeout_overlap_requires_explicit_opt_in():
    release = Event()

    class BlockingAgent:
        name = "blocking"

        def run(self, query, **kwargs):
            release.wait(timeout=1)
            return "late"

    fallback = SequenceAgent("fallback", ["fallback result"])
    flow = LightFlow().step(
        "work",
        agent=BlockingAgent(),
        timeout=0.01,
        fallback_agent=fallback,
        allow_timeout_overlap=True,
    )
    try:
        result = flow.run("go")
    finally:
        release.set()

    assert result.success is True
    assert result.steps[0].used_fallback is True
    assert len(fallback.calls) == 1


def test_timed_executor_is_cleaned_up_on_all_outcomes(monkeypatch):
    shutdown_calls = []

    class TrackingExecutor(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdown_calls.append((wait, cancel_futures))
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(flow_module, "ThreadPoolExecutor", TrackingExecutor)

    successful = SequenceAgent("success", ["done"])
    LightFlow().step("success", agent=successful, timeout=1).run("go")

    raising = SequenceAgent("raising", [RuntimeError("failed")])
    LightFlow().step("raising", agent=raising, timeout=1).run("go")

    release = Event()

    class BlockingAgent:
        name = "blocking"

        def run(self, query, **kwargs):
            release.wait(timeout=1)
            return "late"

    try:
        LightFlow().step("timeout", agent=BlockingAgent(), timeout=0.01).run("go")
    finally:
        release.set()

    assert shutdown_calls == [(True, False), (True, True), (False, True)]


def test_cancellation_is_execution_scoped_and_not_sticky():
    entered = Event()
    release = Event()

    class BlockingAgent:
        name = "blocking"

        def run(self, query, **kwargs):
            entered.set()
            release.wait(timeout=1)
            return "first done"

    second = SequenceAgent("second", ["second done", "second run done"])
    flow = LightFlow().step("first", agent=BlockingAgent()).step(
        "second", agent=second, depends_on=["first"]
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(flow.run, "go", run_id="cancel-me")
        assert entered.wait(timeout=1)
        assert flow.cancel("cancel-me", "operator stop") == 1
        release.set()
        cancelled = future.result(timeout=1)

    later = flow.run("again", run_id="later-run")

    assert cancelled.status == "skipped"
    assert cancelled.success is False
    assert cancelled.steps[-1].status == "skipped"
    assert "operator stop" in cancelled.error
    assert later.success is True
    assert len(second.calls) == 1


def test_targeted_cancellation_does_not_cancel_sibling_run():
    started = Barrier(3)
    release = Event()

    class BlockingAgent:
        name = "blocking"

        def run(self, query, **kwargs):
            started.wait(timeout=1)
            release.wait(timeout=1)
            return f"first:{query}"

    second = SequenceAgent("second", ["sibling completed"])
    flow = LightFlow().step("first", agent=BlockingAgent()).step(
        "second", agent=second, depends_on=["first"]
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelled_future = executor.submit(flow.run, "a", run_id="run-a")
        sibling_future = executor.submit(flow.run, "b", run_id="run-b")
        started.wait(timeout=1)
        assert flow.cancel("run-a", "cancel only a") == 1
        release.set()
        cancelled = cancelled_future.result(timeout=1)
        sibling = sibling_future.result(timeout=1)

    assert cancelled.status == "skipped"
    assert sibling.success is True
    assert sibling.content == "sibling completed"
    assert len(second.calls) == 1


def test_cancel_without_active_execution_is_a_noop():
    agent = SequenceAgent("worker", ["done"])
    flow = LightFlow().step("work", agent=agent)

    assert flow.cancel() == 0
    assert flow.run("go").success is True


def test_external_token_and_step_idempotency_key_reach_agent():
    token = CancellationToken(run_id="propagate")
    agent = SequenceAgent("worker", ["done"])

    result = LightFlow().step("work", agent=agent).run(
        "go",
        run_id="propagate",
        cancellation_token=token,
    )

    kwargs = agent.calls[0]["kwargs"]
    assert kwargs["cancellation_token"] is token
    assert kwargs["idempotency_key"] == "propagate:work"
    assert result.steps[0].idempotency_key == "propagate:work"


def test_lightagent_pre_cancelled_token_fails_before_model_request():
    token = CancellationToken(run_id="agent-run")
    token.cancel("parent workflow stopped")
    agent = LightAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        auto_discover_skills=False,
    )

    result = agent.run(
        "must not reach provider",
        result_format="object",
        trace=True,
        cancellation_token=token,
        idempotency_key="flow:step",
    )

    assert result.error.startswith("[LA-CANCELLED]")
    assert any(
        event["type"] == "run_start"
        and event["data"]["idempotency_key"] == "flow:step"
        for event in result.trace
    )


def test_repeated_run_id_returns_checkpoint_without_side_effects():
    agent = SequenceAgent("worker", ["once"])
    flow = LightFlow().step("work", agent=agent)

    first = flow.run("go", run_id="same-run")
    repeated = flow.run("different input", run_id="same-run")

    assert first.success is True
    assert repeated.success is True
    assert repeated.idempotent_replay is True
    assert repeated.content == "once"
    assert len(agent.calls) == 1


def test_job_idempotency_and_cancellation_token_propagation():
    async def scenario():
        runtime = AgentRuntime()
        runtime.open_session()
        received = []

        async def work(*, cancellation_token):
            received.append(cancellation_token)
            await asyncio.sleep(0)
            return "done"

        first = runtime.jobs.start("work", work, idempotency_key="job-once")
        duplicate = runtime.jobs.start("duplicate", work, idempotency_key="job-once")
        completed = await runtime.jobs.wait(first.job_id)
        return first, duplicate, completed, received

    first, duplicate, completed, received = asyncio.run(scenario())

    assert duplicate.job_id == first.job_id
    assert completed.result == "done"
    assert completed.idempotency_key == "job-once"
    assert len(received) == 1
    assert received[0].token_id == completed.cancellation_token_id


def test_subagent_parent_cancellation_propagates_cooperatively():
    async def scenario():
        runtime = AgentRuntime()
        runtime.open_session()
        started = asyncio.Event()
        release = asyncio.Event()

        class Child:
            name = "child"

            async def arun(self, query, *, cancellation_token, idempotency_key=None):
                started.set()
                await release.wait()
                return cancellation_token.cancelled, idempotency_key

        record = runtime.subagents.register(Child())
        parent = CancellationToken(run_id="parent")
        task = asyncio.create_task(runtime.subagents.run(
            record.agent_id,
            "go",
            cancellation_token=parent,
            idempotency_key="child-once",
        ))
        await started.wait()
        parent.cancel("parent stopped")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runtime.subagents.tree()[0]

    record = asyncio.run(scenario())

    assert record["status"] == "cancelled"
