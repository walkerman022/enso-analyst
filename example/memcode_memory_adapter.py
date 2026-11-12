#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optional Memcode v2 memory adapter for LightAgent.

The adapter accepts an injected ``MemcodeV2Client``-compatible object so
LightAgent does not gain a mandatory SDK dependency or perform network work at
import time. Applications remain responsible for credentials and for mapping
each LightAgent user to a pre-provisioned Memcode space.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

_RESERVED_ATTRIBUTION_FIELDS = {
    "integration_id",
    "integration_channel",
    "attribution_status",
    "attribution_basis",
}


class MemcodeMemoryAdapter:
    """Bridge LightAgent's MemoryProtocol to a tenant-bound Memcode v2 client."""

    def __init__(
            self,
            client: Any,
            *,
            space_id_for_user: Callable[[str], str],
            actor_id_for_user: Callable[[str], str] | None = None,
            agent_name: str = "lightagent",
            top_k: int = 5,
    ):
        if not callable(space_id_for_user):
            raise TypeError("space_id_for_user must be callable")
        if actor_id_for_user is not None and not callable(actor_id_for_user):
            raise TypeError("actor_id_for_user must be callable")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        self.client = client
        self.space_id_for_user = space_id_for_user
        self.actor_id_for_user = actor_id_for_user or (lambda user_id: user_id)
        self.agent_name = str(agent_name)
        self.top_k = top_k

    def store(
            self,
            data: str,
            user_id: str,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a durable Memcode ingest for one explicitly scoped user."""
        memory_user_id = self._required("user_id", user_id)
        content = self._required("data", data)
        space_id = self._space_id(memory_user_id)
        actor_id = self._actor_id(memory_user_id)
        record_metadata = {
            "user_id": memory_user_id,
            "source": "user",
            "scope": "user",
            "agent_name": self.agent_name,
        }
        if metadata:
            reserved = _RESERVED_ATTRIBUTION_FIELDS.intersection(metadata)
            if reserved:
                fields = ", ".join(sorted(reserved))
                raise ValueError(f"Memcode attribution metadata is server-owned: {fields}")
            record_metadata.update(metadata)
        record_metadata["user_id"] = memory_user_id
        record_metadata.setdefault("source", "user")
        record_metadata.setdefault("scope", "user")
        record_metadata.setdefault("agent_name", self.agent_name)

        result = self.client.ingest(
            space_id=space_id,
            actor_id=actor_id,
            content=content,
            idempotency_key=self._idempotency_key(
                user_id=memory_user_id,
                space_id=space_id,
                content=content,
                metadata=record_metadata,
            ),
            metadata=record_metadata,
            tags=self._tags(record_metadata),
        )
        return {
            "stored": True,
            "user_id": memory_user_id,
            "space_id": space_id,
            "job_id": self._value(result, "id", "job_id"),
            "status": self._value(result, "status"),
        }

    def retrieve(self, query: str, user_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return only Memcode results attributed to the requested user/space."""
        memory_user_id = self._required("user_id", user_id)
        normalized_query = self._required("query", query)
        space_id = self._space_id(memory_user_id)
        actor_id = self._actor_id(memory_user_id)
        response = self.client.search(
            context_space_id=space_id,
            actor_id=actor_id,
            query=normalized_query,
            scope="context_only",
            mode="memories",
            top_k=self.top_k,
        )

        results = []
        for item in self._value(response, "results") or []:
            content = self._value(item, "content", "memory", "text")
            metadata = self._value(item, "metadata")
            item_space = self._value(self._value(item, "space"), "id")
            if (
                content is None
                or not isinstance(metadata, dict)
                or str(metadata.get("user_id", "")) != memory_user_id
                or str(item_space or "") != space_id
            ):
                continue
            results.append({
                "memory": str(content),
                "score": self._value(item, "score"),
                "user_id": memory_user_id,
                "metadata": dict(metadata),
            })
        return {"results": results}

    def _space_id(self, user_id: str) -> str:
        return self._required("space_id", self.space_id_for_user(user_id))

    def _actor_id(self, user_id: str) -> str:
        return self._required("actor_id", self.actor_id_for_user(user_id))

    @staticmethod
    def _required(name: str, value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{name} is required")
        return normalized

    @staticmethod
    def _idempotency_key(
            *,
            user_id: str,
            space_id: str,
            content: str,
            metadata: dict[str, Any],
    ) -> str:
        body = json.dumps(
            {
                "user_id": user_id,
                "space_id": space_id,
                "content": content,
                "metadata": metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return f"lightagent:{hashlib.sha256(body).hexdigest()}"

    @staticmethod
    def _tags(metadata: dict[str, Any]) -> list[str]:
        tags = ["integration:lightagent"]
        for name in ("source", "scope", "agent_name"):
            value = metadata.get(name)
            if value:
                tags.append(f"{name}:{value}")
        return tags

    @staticmethod
    def _value(item: Any, *names: str) -> Any:
        if item is None:
            return None
        if isinstance(item, dict):
            for name in names:
                if name in item:
                    return item[name]
            return None
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)
        return None


def build_agent(memory: MemcodeMemoryAdapter) -> Any:
    """Construct a sample agent with fail-closed memory policy defaults."""
    from LightAgent import LightAgent, MemoryPolicy

    return LightAgent(
        name=memory.agent_name,
        role="You are LightAgent with optional Memcode long-term memory.",
        model="deepseek-chat",
        api_key="your_model_api_key",
        base_url="your_model_base_url",
        memory=memory,
        memory_policy=MemoryPolicy(
            namespace="demo",
            allow_unattributed_results=False,
            allowed_sources=("user",),
            allowed_scopes=("user",),
            allowed_agent_names=(memory.agent_name,),
        ),
        tree_of_thought=False,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Create a MemcodeV2Client and a per-user space resolver, then pass the "
        "adapter to build_agent(). See docs/memcode_memory_adapter.md."
    )
