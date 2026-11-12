import importlib.util
from copy import deepcopy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_example_module():
    path = Path(__file__).resolve().parents[1] / "example" / "memcode_memory_adapter.py"
    spec = importlib.util.spec_from_file_location("memcode_memory_adapter_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeMemcodeClient:
    def __init__(self):
        self.ingest_calls = []
        self.search_calls = []
        self.search_results = []
        self.error = None

    def ingest(self, **payload):
        if self.error:
            raise self.error
        self.ingest_calls.append(payload)
        return SimpleNamespace(id="job-1", status="queued")

    def search(self, **payload):
        if self.error:
            raise self.error
        self.search_calls.append(payload)
        return SimpleNamespace(results=self.search_results)


def make_adapter(client):
    module = load_example_module()
    return module.MemcodeMemoryAdapter(
        client,
        space_id_for_user=lambda user_id: {
            "tenant:alice": "space-alice",
            "tenant:bob": "space-bob",
        }[user_id],
        actor_id_for_user=lambda user_id: f"actor:{user_id}",
        agent_name="travel-agent",
        top_k=3,
    )


def test_store_preserves_policy_metadata_and_explicit_scope():
    client = FakeMemcodeClient()
    adapter = make_adapter(client)

    result = adapter.store(
        "Alice prefers quiet beach towns",
        "tenant:alice",
        metadata={"source": "user", "scope": "user", "trace_id": "trace-1"},
    )

    call = client.ingest_calls[0]
    assert result == {
        "stored": True,
        "user_id": "tenant:alice",
        "space_id": "space-alice",
        "job_id": "job-1",
        "status": "queued",
    }
    assert call["space_id"] == "space-alice"
    assert call["actor_id"] == "actor:tenant:alice"
    assert call["metadata"]["user_id"] == "tenant:alice"
    assert call["metadata"]["agent_name"] == "travel-agent"
    assert call["metadata"]["trace_id"] == "trace-1"
    assert call["idempotency_key"].startswith("lightagent:")


def test_store_rejects_server_owned_attribution_metadata():
    client = FakeMemcodeClient()
    adapter = make_adapter(client)

    with pytest.raises(ValueError, match="server-owned"):
        adapter.store(
            "remember this",
            "tenant:alice",
            metadata={"integration_id": "spoofed"},
        )
    assert client.ingest_calls == []


def test_retrieve_filters_cross_user_cross_space_and_malformed_results():
    client = FakeMemcodeClient()
    client.search_results = [
        SimpleNamespace(
            content="Alice prefers quiet beach towns",
            score=0.91,
            metadata={"user_id": "tenant:alice", "source": "user", "scope": "user"},
            space=SimpleNamespace(id="space-alice"),
        ),
        SimpleNamespace(
            content="Bob's private preference",
            score=0.99,
            metadata={"user_id": "tenant:bob", "source": "user", "scope": "user"},
            space=SimpleNamespace(id="space-alice"),
        ),
        SimpleNamespace(
            content="Wrong space",
            score=0.98,
            metadata={"user_id": "tenant:alice", "source": "user", "scope": "user"},
            space=SimpleNamespace(id="space-bob"),
        ),
        SimpleNamespace(content="Missing provenance", score=0.97, metadata={}, space=None),
        {"score": 0.5, "metadata": {"user_id": "tenant:alice"}},
    ]
    adapter = make_adapter(client)

    response = adapter.retrieve("quiet beach", "tenant:alice")

    assert client.search_calls == [{
        "context_space_id": "space-alice",
        "actor_id": "actor:tenant:alice",
        "query": "quiet beach",
        "scope": "context_only",
        "mode": "memories",
        "top_k": 3,
    }]
    assert response["results"] == [{
        "memory": "Alice prefers quiet beach towns",
        "score": 0.91,
        "user_id": "tenant:alice",
        "metadata": {
            "user_id": "tenant:alice",
            "source": "user",
            "scope": "user",
        },
    }]


def test_empty_scope_mapping_fails_closed_before_provider_call():
    module = load_example_module()
    client = FakeMemcodeClient()
    adapter = module.MemcodeMemoryAdapter(
        client,
        space_id_for_user=lambda _user_id: "",
    )

    with pytest.raises(ValueError, match="space_id is required"):
        adapter.retrieve("hello", "tenant:alice")
    assert client.search_calls == []


def test_provider_errors_propagate_without_fallback():
    client = FakeMemcodeClient()
    client.error = RuntimeError("provider unavailable")
    adapter = make_adapter(client)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        adapter.retrieve("hello", "tenant:alice")


class RoundTripMemcodeClient(FakeMemcodeClient):
    def search(self, **payload):
        self.search_calls.append(payload)
        return SimpleNamespace(results=[
            SimpleNamespace(
                content=call["content"],
                metadata=call["metadata"],
                space=SimpleNamespace(id=call["space_id"]),
                score=1.0,
            )
            for call in self.ingest_calls
        ])


class StaticCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **params):
        self.calls.append(deepcopy(params))
        message = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize("agent_name", ["lightagent", "travel-agent"])
def test_build_agent_memory_roundtrip_preserves_identity_scope_and_hooks(agent_name):
    module = load_example_module()
    client = RoundTripMemcodeClient()
    memory = module.MemcodeMemoryAdapter(
        client,
        space_id_for_user={"demo:alice": "space-alice", "demo:bob": "space-bob"}.__getitem__,
        agent_name=agent_name,
    )
    agent = module.build_agent(memory)
    completions = StaticCompletions()
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    phases = []

    def observe(context):
        if context.phase in {"before_memory_retrieve", "after_memory_retrieve"}:
            phases.append(context.phase)

    agent.hooks.hooks.append(observe)
    assert agent.name == agent_name
    assert agent.memory_policy.allowed_agent_names == (agent_name,)

    agent.run("Alice prefers quiet beach towns", user_id="alice")
    stored = client.ingest_calls[0]
    assert stored["space_id"] == "space-alice"
    assert stored["metadata"]["user_id"] == "demo:alice"
    assert stored["metadata"]["agent_name"] == agent_name

    result = agent.run("What does Alice prefer?", user_id="alice", result_format="object", trace=True)
    assert result.error is None
    prompt = completions.calls[-1]["messages"][-1]["content"]
    assert "Alice prefers quiet beach towns" in prompt
    filtered = next(event for event in result.trace if event["type"] == "memory_retrieve_filter")
    assert filtered["data"]["allowed_count"] >= 1

    bob = agent.run("What do I prefer?", user_id="bob", result_format="object", trace=True)
    assert bob.error is None
    assert "Alice prefers quiet beach towns" not in completions.calls[-1]["messages"][-1]["content"]
    assert client.search_calls[-1]["context_space_id"] == "space-bob"
    assert client.ingest_calls[-1]["metadata"]["user_id"] == "demo:bob"
    assert phases == ["before_memory_retrieve", "after_memory_retrieve"] * 3
