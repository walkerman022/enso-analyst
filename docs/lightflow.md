## LightFlow

LightAgent v0.8.0 introduces `LightFlow`, a lightweight deterministic workflow
runner for chaining multiple `LightAgent` instances into explicit steps.

`LightFlow` is intentionally small: it provides DAG-style dependencies, step
input/output passing, step retries, step status tracking, checkpointable run
records, and flow-level trace events without adding a heavy orchestration
dependency.

See `example/10.lightflow.py` for a complete runnable example.

### Basic Usage

```python
from LightAgent import LightAgent, LightFlow

research_agent = LightAgent(
    name="ResearchAgent",
    model="gpt-4.1",
    api_key="your_api_key",
    base_url="your_base_url",
)

writer_agent = LightAgent(
    name="WriterAgent",
    model="gpt-4.1",
    api_key="your_api_key",
    base_url="your_base_url",
)

flow = (
    LightFlow()
    .step("research", agent=research_agent)
    .step("write", agent=writer_agent, depends_on=["research"])
)

result = flow.run("Analyze this company", trace=True)

print(result.content)
print(result.success)
print(result.trace)
```

When a step depends on previous steps and does not define a custom query,
LightFlow appends the dependency outputs to the original input.

### Custom Step Input

Use a callable `query` when a step needs precise control over its prompt.

```python
flow = (
    LightFlow()
    .step("research", agent=research_agent)
    .step(
        "write",
        agent=writer_agent,
        depends_on=["research"],
        query=lambda context: f"Write a concise report from: {context['outputs']['research']}",
    )
)
```

The callable receives a context dictionary:

| Key | Meaning |
| --- | --- |
| `input` | The original flow input. |
| `outputs` | Mapping of completed step name to string output. |
| `steps` | Mapping of completed step name to `LightFlowStepResult`. |

### Step Retries

Each step can retry when the underlying agent returns a structured error or
raises an ordinary `Exception`. Raised exception details are not copied into
workflow results; the exception type and a stable flow error code are retained.

```python
flow.step("research", agent=research_agent, max_retry=2)
```

Retries are step-local. If a step still fails after its retries, the flow stops
and returns a `LightFlowResult` with `success == False`.

### Step Status And Controls

Each `LightFlowStepResult` includes a `status` value:

| Status | Meaning |
| --- | --- |
| `pending` | Step has not started yet. |
| `running` | Step is currently executing. |
| `success` | Step completed successfully. |
| `failed` | Step failed after retries and fallback handling. |
| `skipped` | Step did not run because the flow was cancelled or a dependency failed. |
| `waiting_approval` | Step requires human approval before execution. |

Steps support timeout, cancellation, fallback agents, and approval handlers:

```python
flow.step(
    "review",
    agent=review_agent,
    depends_on=["draft"],
    timeout=30,
    fallback_agent=fallback_review_agent,
    requires_approval=True,
    approval_handler=lambda step, context: True,
)
```

Step timeouts are soft wall-clock bounds because a Python worker thread cannot
be forcefully terminated safely. By default, a timed-out call fails the step
without starting a retry or fallback while that call may still be running. If
the operation is idempotent and overlapping execution is acceptable, opt in
explicitly:

```python
flow.step(
    "review",
    agent=review_agent,
    timeout=30,
    max_retry=2,
    fallback_agent=fallback_review_agent,
    allow_timeout_overlap=True,
)
```

With this option enabled, timeout retries and fallback have at-least-once
semantics and may overlap. A terminable external SandboxProvider should be used
when strict process termination is required.

### Cancellation

`flow.cancel()` cancels every execution currently active on that `LightFlow`
instance. Pass a `run_id` to target one active run. Calling it when no execution
is active is a no-op and does not poison future `run()`, `resume()`, or
`rerun_step()` calls.

For independent ownership, pass an explicit cooperative token:

```python
from LightAgent import CancellationToken

token = CancellationToken(run_id="report-001")
result = flow.run(
    "Analyze this company",
    run_id="report-001",
    cancellation_token=token,
)

# From another thread or controller:
token.cancel("operator stopped the workflow")
```

Cancellation is checked before each step and propagated to agents that accept
a `cancellation_token` keyword. It does not forcefully stop a model call already
in progress.

v0.9.6 approval handlers may also return `ApprovalDecision.approve()`,
`reject()`, `edit({"query": "..."})`, or `respond("...")`. Boolean handlers
remain compatible.

Use `flow.validate()` to inspect workflow structure before execution. Unknown
dependencies and cycles are reported as errors. Isolated steps are reported as
warnings so they can be reviewed without blocking simple workflows.

### Result Formats

The default result is a `LightFlowResult` object:

```python
result = flow.run("Analyze this company")
print(result.content)
print(result.steps[0].content)
print(result.error)
```

You can also request a string or dictionary:

```python
text = flow.run("Analyze this company", result_format="str")
data = flow.run("Analyze this company", result_format="dict")
```

### Trace Events

Pass `trace=True` to collect flow-level events:

| Event | Meaning |
| --- | --- |
| `flow_start` | The flow started. |
| `step_start` | A step started. |
| `step_end` | A step completed or failed. |
| `flow_end` | The flow completed or stopped on failure. |
| `approval_pending` | A step is waiting for a durable decision. |
| `approval_approve` / `approval_reject` / `approval_edit` / `approval_respond` | A step review decision was applied. |

Each step also preserves the underlying agent trace when the agent returns a
structured `RunResult`.

`step_end` events include status, attempts, retry count, error reason, duration,
input summary, output summary, and whether a fallback agent was used.

### Persistent Runs

Use `JsonLightFlowStore` to persist checkpoints and inspect workflow run
records:

```python
from LightAgent import JsonLightFlowStore, LightFlow

store = JsonLightFlowStore(".lightflow_runs")

flow = (
    LightFlow(store=store)
    .step("research", agent=research_agent)
    .step("write", agent=writer_agent, depends_on=["research"])
)

result = flow.run("Analyze this company", run_id="report-001")

if not result.success:
    result = flow.resume("report-001")
```

For a step waiting on human review, persist a decision and resume:

```python
from LightAgent import ApprovalDecision

waiting = flow.run("Publish the report", run_id="report-001")
request_id = waiting.steps[0].approval_request_id

flow.approve(
    "report-001",
    "publish",
    ApprovalDecision.approve(reviewer_id="editor"),
)
result = flow.resume("report-001")
```

The request ID and decision are stored in the JSON checkpoint, so a new
application process can rebuild the same flow definition and resume the run.

For a selected step and its downstream dependencies:

```python
result = flow.rerun_step("report-001", "write")
```

Use `get_run(run_id)` and `list_runs()` to build front-end execution views.

An explicit `run_id` is also the local idempotency boundary. Calling `run()`
again with a run ID already present in the configured store returns the saved
checkpoint with `idempotent_replay == True`; it does not execute agents again.
Use `resume()` or `rerun_step()` when execution is intentional. Each step also
receives a stable `<run_id>:<step_name>` idempotency key when its agent accepts
the `idempotency_key` keyword.

### Current Scope

The v0.10.2 implementation provides execution-scoped cancellation, local
run-ID idempotency, and lightweight JSON checkpoints. Production deployments
that need cross-process locking, distributed workers, hard process termination,
or globally consistent idempotency should provide stronger Provider and
run-store implementations around the same contracts.
