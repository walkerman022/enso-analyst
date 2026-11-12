## Memcode Memory Adapter Example

LightAgent can use Memcode as an optional long-term-memory backend through its
existing `store(data, user_id)` / `retrieve(query, user_id)` protocol. The
example adapter is dependency-free at import time: applications inject a
`MemcodeV2Client` from the optional `memcode-sdk` package, while tests use a
fake client and make no network requests.

### Install and configure

```bash
pip install memcode-sdk
```

Keep credentials outside source control. Create the key from the
[Memcode API-key dashboard](https://app.memcode.in/dashboard?section=api-keys&integration=lightagent) with **LightAgent**
selected under integration attribution:

```bash
export MEMCODE_API_URL=https://memory.memcode.in
export MEMCODE_API_KEY=your-integration-key
```

Each LightAgent user must map to a pre-provisioned Memcode space. The adapter
does not invent, share, or fall back to a default space:

```python
from memcode_sdk import MemcodeV2Client

from LightAgent import LightAgent, MemoryPolicy
from example.memcode_memory_adapter import MemcodeMemoryAdapter


client = MemcodeV2Client()
spaces = {
    "tenant-a:alice": "space-for-alice",
    "tenant-a:bob": "space-for-bob",
}

memory = MemcodeMemoryAdapter(
    client,
    space_id_for_user=spaces.__getitem__,
    actor_id_for_user=lambda user_id: user_id,
    agent_name="support-agent",
)

agent = LightAgent(
    name="support-agent",
    model="gpt-4.1",
    api_key="your_model_api_key",
    base_url="your_model_base_url",
    memory=memory,
    memory_policy=MemoryPolicy(
        namespace="tenant-a",
        allow_unattributed_results=False,
        allowed_sources=("user",),
        allowed_scopes=("user",),
        allowed_agent_names=("support-agent",),
    ),
)
```

### Security and lifecycle behavior

- Keep the agent name, adapter provenance name, and policy allowlist aligned.
  The example's `build_agent(memory)` uses `memory.agent_name` for all three;
  its space resolver receives namespaced IDs such as `demo:alice`.
- The application owns API-key or OAuth storage. The adapter never reads a
  credential at import time and never returns credentials to the agent.
- `space_id_for_user` and `actor_id_for_user` are evaluated for every call.
  Empty mappings fail before any provider request.
- Writes preserve LightAgent provenance metadata. Server-owned integration
  attribution fields are rejected instead of being forwarded or spoofed.
- Reads use `context_only` scope and then require both the requested Memcode
  space and exact `metadata.user_id`; missing or mismatched provenance is
  dropped before `MemoryPolicy` sees it.
- Provider, authentication, and network failures propagate to the caller. The
  adapter does not fall back to another user, tenant, or global search.
- Memcode ingestion is durable and asynchronous. `store()` returns the job ID
  and initial status; applications that require ready-before-read semantics
  should poll with `client.get_ingest_status(job_id)`.
- Retention is configured in Memcode. LightAgent's current memory protocol has
  no delete method, and the v2 SDK adapter deliberately does not simulate one;
  use the authorized Memcode lifecycle API or console for deletion/forgetting.

Memcode binds the `lightagent` identity when the key is issued. A generic key
still works, but its requests are counted as generic direct API usage. The SDK
never supplies attribution itself. Do not add
`integration_id`, `integration_channel`, `attribution_status`, or
`attribution_basis` to memory metadata.
