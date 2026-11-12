#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆：只存"偏好"这一类，不存实时数据。

为什么只存偏好：
气象数据是实时查的，存下来就会过期——存"昨天 ONI 是 2.46"没有意义，
用户问的时候查一次更准，而且过期数据被模型当成当前值引用比没有数据更危险。
真正跨会话有价值的是"用户长期关注什么"：区域偏好、分型偏好、语言、单位。

存储：单表 preferences，键值 + 更新时间 + 访问计数。
检索：全量读进系统提示（量级只有几条到几十条，不需要向量检索）。
冲突：同键覆盖，保留 updated_at，历史不追溯。
遗忘：记录 hits / last_used_at，长期不用的可以在注入时截断（按 last_used_at 排序取前 N）。
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "memory.db"

MAX_INJECT = 12          # 注入系统提示的最大条数
MAX_VALUE_CHARS = 200    # 单条值长度上限，防止把长文档塞进来


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT
        )
        """
    )
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def remember(key: str, value: str) -> dict[str, Any]:
    """写入或覆盖一条偏好。同键覆盖，不保留历史。"""
    key = (key or "").strip()
    value = (value or "").strip()[:MAX_VALUE_CHARS]
    if not key or not value:
        raise ValueError("key 和 value 都不能为空")
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO preferences(key, value, updated_at, hits, last_used_at)
            VALUES(?, ?, ?, 0, NULL)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, _now()),
        )
    _invalidate_cache("prefs")
    return {"key": key, "value": value, "action": "saved"}


def forget(key: str) -> dict[str, Any]:
    with closing(_connect()) as connection, connection:
        cursor = connection.execute("DELETE FROM preferences WHERE key = ?", ((key or "").strip(),))
    if cursor.rowcount:
        _invalidate_cache("prefs")
    return {"key": key, "action": "deleted" if cursor.rowcount else "not_found"}


def _invalidate_cache(tag: str) -> None:
    """偏好变了，提示缓存里带该标签的条目必须失效。"""
    try:
        from weather_agent import prompt_cache

        prompt_cache.invalidate(tag)
    except Exception:
        pass


def recall(key: str | None = None) -> Any:
    """按 key 精确查询；不传 key 返回全部。"""
    with closing(_connect()) as connection, connection:
        if key:
            row = connection.execute("SELECT * FROM preferences WHERE key = ?", (key.strip(),)).fetchone()
            return dict(row) if row else None
        rows = connection.execute(
            "SELECT * FROM preferences ORDER BY COALESCE(last_used_at, updated_at) DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def touch(keys: list[str]) -> None:
    """注入之后记一次使用，用于后续按活跃度排序。"""
    if not keys:
        return
    with closing(_connect()) as connection, connection:
        for key in keys:
            connection.execute(
                "UPDATE preferences SET hits = hits + 1, last_used_at = ? WHERE key = ?",
                (_now(), key),
            )


def preference_block(limit: int = MAX_INJECT) -> str:
    """生成注入系统提示的片段。没有偏好时返回空串。"""
    rows = recall()[:limit]
    if not rows:
        return ""
    lines = [f"- {row['key']}: {row['value']}" for row in rows]
    touch([row["key"] for row in rows])
    return "## 已知用户偏好（跨会话记忆，可直接采用）\n" + "\n".join(lines)


def version() -> str:
    """偏好版本：条数 + 最新更新时间。用于提示缓存的键，偏好一变键就变。"""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), '') AS latest FROM preferences"
        ).fetchone()
    return f"{row['n']}@{row['latest']}" if row else "0@"
