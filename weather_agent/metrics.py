#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行度量：每次问答结束后落一条记录，供聚合报表使用。

记什么：模式、问题、模型调用次数、工具调用次数、token、耗时、护栏问题数、引用数、是否降级。
为什么记这些：它们能回答三个问题——贵不贵（token/调用次数）、慢不慢（P50/P95 延迟）、
靠不靠谱（护栏触发率）。

不记完整回答：体积大且含用户内容；只存问题前 60 字，用于定位。
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "metrics.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            mode TEXT NOT NULL,
            question TEXT,
            latency_ms INTEGER,
            model_calls INTEGER,
            tool_calls INTEGER,
            tokens INTEGER,
            guardrail_issues INTEGER,
            citations INTEGER,
            degraded INTEGER,
            success INTEGER,
            cache_hit INTEGER DEFAULT 0
        )
        """
    )
    # 兼容旧库：缺列就补上
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    if "cache_hit" not in columns:
        connection.execute("ALTER TABLE runs ADD COLUMN cache_hit INTEGER DEFAULT 0")
        connection.commit()
    return connection


def snapshot(agent: Any) -> dict[str, float]:
    """读当前累计用量，用于算增量。"""
    try:
        usage = agent.runtime.budget.usage
        # 注意：model_calls 每次 run 会重置，是"本次"的计数；
        # tool_calls 与 tokens 是累计值，要取增量。这是框架的实际行为，已实测确认。
        return {
            "model_calls": float(usage.model_calls),
            "tool_calls": float(usage.tool_calls),
            "tokens": float(usage.tokens or 0),
        }
    except Exception:
        return {"model_calls": 0.0, "tool_calls": 0.0, "tokens": 0.0}


def record(
        *,
        mode: str,
        question: str,
        before: dict[str, float],
        after: dict[str, float],
        latency_ms: int,
        guardrail_issues: int = 0,
        citations: int = 0,
        degraded: bool = False,
        success: bool = True,
        cache_hit: bool = False,
        db_path: Path | None = None,
) -> None:
    """写一条运行记录。失败不影响主流程。"""
    global DB_PATH
    original = DB_PATH
    if db_path is not None:
        DB_PATH = db_path
    try:
        with closing(_connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO runs(ts, mode, question, latency_ms, model_calls, tool_calls,
                                 tokens, guardrail_issues, citations, degraded, success, cache_hit)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    mode,
                    (question or "")[:60],
                    int(latency_ms),
                    int(after.get("model_calls", 0)),
                    int(after.get("tool_calls", 0) - before.get("tool_calls", 0)),
                    int(after.get("tokens", 0) - before.get("tokens", 0)),
                    int(guardrail_issues),
                    int(citations),
                    int(bool(degraded)),
                    int(bool(success)),
                    int(bool(cache_hit)),
                ),
            )
    except Exception:
        # 度量失败不能影响回答
        pass
    finally:
        DB_PATH = original


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return float(ordered[index])


def summary(limit: int = 500, db_path: Path | None = None) -> dict[str, Any]:
    """聚合最近若干条记录。"""
    global DB_PATH
    original = DB_PATH
    if db_path is not None:
        DB_PATH = db_path
    try:
        with closing(_connect()) as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()]
    finally:
        DB_PATH = original

    if not rows:
        return {"count": 0}

    latencies = [row["latency_ms"] for row in rows]
    return {
        "count": len(rows),
        "success_rate": round(mean([row["success"] for row in rows]), 3),
        "latency_p50_ms": int(_percentile(latencies, 0.5)),
        "latency_p95_ms": int(_percentile(latencies, 0.95)),
        "latency_avg_ms": int(mean(latencies)),
        "model_calls_avg": round(mean([row["model_calls"] for row in rows]), 2),
        "tool_calls_avg": round(mean([row["tool_calls"] for row in rows]), 2),
        "tokens_avg": round(mean([row["tokens"] for row in rows]), 1),
        "guardrail_rate": round(mean([1 if row["guardrail_issues"] else 0 for row in rows]), 3),
        "citation_rate": round(mean([1 if row["citations"] else 0 for row in rows]), 3),
        "degraded_rate": round(mean([row["degraded"] for row in rows]), 3),
        "cache_hit_rate": round(mean([row.get("cache_hit", 0) or 0 for row in rows]), 3),
    }


def timed_call(agent: Any, question: str, *, mode: str, scope: str = "default",
               recorder=None, session_id: str | None = None) -> tuple[str, int]:
    """跑一次问答并记录度量。返回 (回答, 耗时毫秒)。

    model_calls 取本次 run 的计数，tool_calls/tokens 取增量——两者语义不同，见 snapshot 注释。

    recorder 默认用模块内的 record；测试时可替换成收集器。
    """
    from weather_agent.number_check import verify

    before = snapshot(agent)
    started = time.perf_counter()
    text = agent.run(question, session_id=session_id) if session_id else agent.run(question)
    text = text if isinstance(text, str) else str(text)
    latency_ms = int((time.perf_counter() - started) * 1000)
    after = snapshot(agent)

    try:
        checked = verify(text, scope=scope)
        issues = len(checked.unverified) + len(checked.mismatched) + len(checked.bad_citations)
        citations = len(checked.citations)
    except Exception:
        issues, citations = 0, 0

    payload = {
        "mode": mode,
        "question": question,
        "before": before,
        "after": after,
        "latency_ms": latency_ms,
        "guardrail_issues": issues,
        "citations": citations,
        "degraded": after.get("tokens", 0) == 0,
        "success": not text.startswith("[LA-"),
    }
    (recorder or record)(**payload)
    return text, latency_ms
