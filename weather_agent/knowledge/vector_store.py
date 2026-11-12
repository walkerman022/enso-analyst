#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""向量库：Qdrant 本地模式（embedded，不需要起服务）。

为什么用 Qdrant 而不是自己算：
语料只有几十片时，numpy 点积确实够；但"用向量库"能带来三件实际好处——
持久化、按 payload 过滤、以及后续换更大的语料时不用改检索代码。
本地模式把数据落在一个目录里，不需要 Docker、不需要端口。

集合名固定 enso_kb，向量维度取 embedding 的实际维度。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

QDRANT_PATH = Path(__file__).resolve().parent.parent / "data" / "qdrant"
COLLECTION = "enso_kb"

# Qdrant 的点 id 只接受整数或 UUID，我们的片段 id 是可读字符串（def-oni），
# 所以用 uuid5 做确定性映射，原始 id 放在 payload 里，检索结果仍返回可读 id。
_ID_NAMESPACE = uuid.UUID("6f6f6e2d-7a3a-5b1e-9c4d-2b7c1e0a9d11")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _client():
    from qdrant_client import QdrantClient

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(dim: int, reset: bool = False) -> None:
    from qdrant_client.models import Distance, VectorParams

    client = _client()
    try:
        exists = client.collection_exists(COLLECTION)
        if reset and exists:
            client.delete_collection(COLLECTION)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    finally:
        client.close()


def upsert(points: list[dict[str, Any]]) -> int:
    """points: [{id, vector, payload}]"""
    if not points:
        return 0
    from qdrant_client.models import PointStruct

    client = _client()
    try:
        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=point_id(str(item["id"])),
                    vector=item["vector"],
                    payload={**item.get("payload", {}), "id": str(item["id"])},
                )
                for item in points
            ],
        )
    finally:
        client.close()
    return len(points)


def search(vector: list[float], top_k: int = 4, score_threshold: float | None = None) -> list[dict[str, Any]]:
    client = _client()
    try:
        if not client.collection_exists(COLLECTION):
            return []
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=max(1, int(top_k)),
                score_threshold=score_threshold,
                with_payload=True,
            )
            hits = response.points
        else:  # 兼容旧版客户端
            hits = client.search(
                collection_name=COLLECTION,
                query_vector=vector,
                limit=max(1, int(top_k)),
                score_threshold=score_threshold,
                with_payload=True,
            )
        return [
            {
                "id": str((hit.payload or {}).get("id", hit.id)),
                "score": float(hit.score),
                "payload": dict(hit.payload or {}),
            }
            for hit in hits
        ]
    finally:
        client.close()


def count() -> int:
    client = _client()
    try:
        if not client.collection_exists(COLLECTION):
            return 0
        return int(client.count(COLLECTION).count)
    finally:
        client.close()
