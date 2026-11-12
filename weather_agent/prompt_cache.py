#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提示缓存（应用层）：把"完全相同的提示"对应的答案存下来，避免重复调用模型。

和"提供方前缀缓存"的区别：
- 提供方前缀缓存由服务端做，命中只减价不减次数，我们无法控制，只能靠稳定的 system 前缀去争取；
- 这里的缓存是应用层的，命中直接不调模型，省的是整次调用。
两者互补：前者管"同一段前缀"，后者管"同一个问题"。

缓存键由五部分组成，任一变化都会 miss：
1. 模型名
2. 系统提示（instructions）哈希
3. 完整消息序列（含历史）哈希
4. 工具 schema 哈希——换工具、改描述都会让旧答案失效
5. 环境版本：长期记忆更新时间 + 知识库版本

为什么要把工具 schema 和环境版本放进键里：
只按问题做键是最常见的错误。工具改了、偏好改了、知识库重建了，
问题字面相同但正确答案已经变了，命中旧值就是错误答案。

失效策略：
- TTL（默认 10 分钟）兜住数据源更新；
- 标签失效：记忆写入/删除 → invalidate("prefs")；知识库重建 → invalidate("kb")；
- 版本号进键：即使忘了调失效，键也变了，不会命中脏数据（双保险）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "prompt_cache.db"
DEFAULT_TTL = 600          # 10 分钟：足够覆盖连续追问，又不会让数据太旧
MAX_ENTRIES = 2000         # 防止无限增长


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0,
            last_hit REAL,
            tags TEXT NOT NULL DEFAULT ''
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_expires ON prompt_cache(expires_at)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS prompt_cache_counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
    )
    return connection


def _now() -> float:
    return time.time()


def signature(
        *,
        model: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
        environment: dict[str, Any] | None = None,
) -> str:
    """生成缓存键。所有影响答案的输入都必须进来。"""
    payload = {
        "model": model,
        "instructions": hashlib.sha256((instructions or "").encode("utf-8")).hexdigest(),
        "messages": hashlib.sha256(
            json.dumps(messages or [], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "tools": hashlib.sha256(
            json.dumps(tools_schema or [], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "env": environment or {},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _bump(connection: sqlite3.Connection, name: str, amount: int = 1) -> None:
    connection.execute(
        """
        INSERT INTO prompt_cache_counters(name, value) VALUES(?, ?)
        ON CONFLICT(name) DO UPDATE SET value = value + excluded.value
        """,
        (name, int(amount)),
    )


def get(key: str) -> str | None:
    """命中返回文本并累加计数；过期条目顺手删掉，并记一次 miss。"""
    now = _now()
    with closing(_connect()) as connection, connection:
        row = connection.execute(
            "SELECT value, expires_at FROM prompt_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None or row["expires_at"] <= now:
            if row is not None:
                connection.execute("DELETE FROM prompt_cache WHERE key = ?", (key,))
            _bump(connection, "miss")
            return None
        connection.execute(
            "UPDATE prompt_cache SET hits = hits + 1, last_hit = ? WHERE key = ?", (now, key)
        )
        _bump(connection, "hit")
        return str(row["value"])


def put(
        key: str,
        value: str,
        *,
        ttl: int = DEFAULT_TTL,
        tags: list[str] | None = None,
) -> None:
    now = _now()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO prompt_cache(key, value, created_at, expires_at, hits, last_hit, tags)
            VALUES(?, ?, ?, ?, 0, NULL, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, created_at=excluded.created_at,
                expires_at=excluded.expires_at, tags=excluded.tags
            """,
            (key, value, now, now + int(ttl), ",".join(tags or [])),
        )
        _evict(connection)


def _evict(connection: sqlite3.Connection) -> None:
    """超过上限时删最久未用的条目。"""
    count = connection.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
    if count <= MAX_ENTRIES:
        return
    connection.execute(
        """
        DELETE FROM prompt_cache WHERE key IN (
            SELECT key FROM prompt_cache
            ORDER BY COALESCE(last_hit, created_at) ASC
            LIMIT ?
        )
        """,
        (count - MAX_ENTRIES,),
    )


def invalidate(*tags: str) -> int:
    """按标签失效。tags 为空则清空全部。"""
    with closing(_connect()) as connection, connection:
        if not tags:
            cursor = connection.execute("DELETE FROM prompt_cache")
            _bump(connection, "invalidate_all")
            return cursor.rowcount or 0
        removed = 0
        for tag in tags:
            cursor = connection.execute(
                "DELETE FROM prompt_cache WHERE ',' || tags || ',' LIKE ?", (f"%,{tag},%",)
            )
            removed += cursor.rowcount or 0
        return removed


def cleanup_expired() -> int:
    with closing(_connect()) as connection, connection:
        cursor = connection.execute("DELETE FROM prompt_cache WHERE expires_at <= ?", (_now(),))
        return cursor.rowcount or 0


def stats() -> dict[str, Any]:
    """命中统计：条目数、命中/未命中次数、命中率。"""
    with closing(_connect()) as connection:
        row = connection.execute("SELECT COUNT(*) AS entries FROM prompt_cache").fetchone()
        counters = {
            item["name"]: int(item["value"])
            for item in connection.execute("SELECT name, value FROM prompt_cache_counters").fetchall()
        }
    entries = int(row["entries"]) if row else 0
    hits = counters.get("hit", 0)
    misses = counters.get("miss", 0)
    total = hits + misses
    return {
        "entries": entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _reset_for_tests(db_path: Path) -> None:
    """测试专用：把库指到临时文件。"""
    global DB_PATH
    DB_PATH = db_path
