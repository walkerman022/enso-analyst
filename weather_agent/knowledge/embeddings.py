#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文本向量化：调百炼的 OpenAI 兼容接口，本地缓存结果。

为什么单独一层：
- 向量化按量计费，重建索引时不该重复付费，所以按内容哈希缓存；
- 接口可能失败，上层必须有降级路径（退回词频检索），不能让整个问答挂掉。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from array import array
from contextlib import closing
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from weather_agent.credentials import DEFAULT_ENV_FILE

BATCH = 10          # 单次请求条数，接口对批量有限制
DIM = 1024          # qwen3.7-text-embedding-flash 默认维度

_CACHE_DB = Path(__file__).resolve().parent.parent / "data" / "knowledge.db"


def _load_env() -> dict[str, str] | None:
    path = Path(DEFAULT_ENV_FILE)
    if path.exists():
        load_dotenv(path, override=False)
    key = os.getenv("EMBEDDING_API_KEY")
    base = os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL")
    if not (key and base and model):
        return None
    return {"api_key": key, "base_url": base, "model": model}


def available() -> bool:
    return _load_env() is not None


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}:{text}".encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_CACHE_DB, timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embedding_cache (key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
    )
    return connection


def _read_cache(keys: list[str]) -> dict[str, list[float]]:
    if not keys:
        return {}
    result: dict[str, list[float]] = {}
    placeholders = ",".join("?" for _ in keys)
    with closing(_connect()) as connection:
        rows = connection.execute(
            f"SELECT key, dim, vector FROM embedding_cache WHERE key IN ({placeholders})", keys
        ).fetchall()
    for key, dim, blob in rows:
        values = array("f")
        values.frombytes(blob)
        if len(values) == dim:
            result[key] = list(values)
    return result


def _write_cache(items: dict[str, list[float]]) -> None:
    if not items:
        return
    with closing(_connect()) as connection, connection:
        for key, vector in items.items():
            values = array("f", vector)
            connection.execute(
                "INSERT OR REPLACE INTO embedding_cache(key, dim, vector) VALUES(?, ?, ?)",
                (key, len(vector), values.tobytes()),
            )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化，命中缓存的不再请求。失败抛 RuntimeError，由上层降级。"""
    config = _load_env()
    if config is None:
        raise RuntimeError("未配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL")

    keys = [_cache_key(text, config["model"]) for text in texts]
    cached = _read_cache(keys)
    missing = [(key, text) for key, text in zip(keys, texts) if key not in cached]

    if missing:
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"], timeout=60, max_retries=1)
        for start in range(0, len(missing), BATCH):
            batch = missing[start:start + BATCH]
            response = client.embeddings.create(
                model=config["model"],
                input=[text for _, text in batch],
            )
            fresh = {key: list(item.embedding) for (key, _), item in zip(batch, response.data)}
            _write_cache(fresh)
            cached.update(fresh)

    return [cached[key] for key in keys]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def stats() -> dict[str, Any]:
    with closing(_connect()) as connection:
        row = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
    return {"cached": int(row[0]) if row else 0, "available": available()}
