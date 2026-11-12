#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识片段的存储与检索。

检索方式：中文二元组 + 英文单词，按 IDF 加权打分。
为什么不用向量：语料只有几十到几百个片段，词频检索足够准，且完全确定、可测；
向量检索要多一个模型依赖、多一层不确定，规模没到就用不上。
"""

from __future__ import annotations

import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.db"

# 高频虚词构成的二元组，命中它们不代表相关。不加这张表，"今天午饭吃什么"会匹配到 ONI 卡片。
_STOP = {
    "什么", "怎么", "为什", "么样", "是什", "有什", "如何", "哪些", "请问", "一下",
    "可以", "我们", "你们", "这个", "那个", "就是", "还是", "以及", "而且", "但是",
}

MIN_SCORE = 1.2   # 低于这个分数认为不相关，宁可不给，也不要给错

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ASCII = re.compile(r"[a-z0-9][a-z0-9._+-]*")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(reset: bool = False) -> None:
    with closing(_connect()) as connection, connection:
        if reset:
            connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT,
                text TEXT NOT NULL,
                source_url TEXT,
                source_name TEXT,
                license TEXT,
                layer TEXT,
                fetched_at TEXT
            )
            """
        )


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    """写入片段，同 id 覆盖。"""
    if not chunks:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as connection, connection:
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO chunks(id, doc_id, title, section, text, source_url, source_name, license, layer, fetched_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, section=excluded.section, text=excluded.text,
                    source_url=excluded.source_url, source_name=excluded.source_name,
                    license=excluded.license, layer=excluded.layer, fetched_at=excluded.fetched_at
                """,
                (
                    chunk["id"], chunk.get("doc_id", ""), chunk.get("title", ""), chunk.get("section", ""),
                    chunk["text"], chunk.get("source_url", ""), chunk.get("source_name", ""),
                    chunk.get("license", ""), chunk.get("layer", ""), chunk.get("fetched_at") or now,
                ),
            )
    return len(chunks)


def count_chunks() -> int:
    with closing(_connect()) as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0


def get_chunks(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with closing(_connect()) as connection:
        rows = connection.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
    return [dict(row) for row in rows]


def tokenize(text: str) -> list[str]:
    """中文取二元组，英文按单词。二元组能避免中文分词的依赖。"""
    lowered = (text or "").lower()
    tokens: list[str] = []
    for run in _CJK.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    tokens.extend(_ASCII.findall(lowered))
    return [token for token in tokens if token not in _STOP]


def search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """返回最相关的片段，附带分数与出处。"""
    query_tokens = list(dict.fromkeys(tokenize(query)))
    if not query_tokens:
        return []

    with closing(_connect()) as connection:
        rows = [dict(row) for row in connection.execute("SELECT * FROM chunks").fetchall()]
    if not rows:
        return []

    doc_tokens = [set(tokenize(f"{row['title']} {row['section'] or ''} {row['text']}")) for row in rows]
    total = len(rows)
    idf: dict[str, float] = {}
    for token in query_tokens:
        df = sum(1 for tokens in doc_tokens if token in tokens)
        idf[token] = math.log((total + 1) / (df + 0.5))

    scored: list[tuple[float, int]] = []
    for index, tokens in enumerate(doc_tokens):
        matched = [token for token in query_tokens if token in tokens]
        if not matched:
            continue
        score = sum(idf[token] for token in matched) / math.sqrt(len(tokens) or 1)
        # 标题命中给额外权重：问「ONI 定义」时，标题就叫「ONI 的定义」的卡片应该排最前
        title_tokens = set(tokenize(rows[index]["title"]))
        score += 0.5 * sum(idf[token] for token in matched if token in title_tokens)
        scored.append((score, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        # 砍掉明显不相关的尾部：低于最高分三分之一的不返回，
        # 否则「东部型和中部型」会把「数据为什么滞后」也带出来，白占上下文。
        floor = max(MIN_SCORE, scored[0][0] * 0.33)
        scored = [item for item in scored if item[0] >= floor]
    results: list[dict[str, Any]] = []
    for score, index in scored[: max(1, int(top_k))]:
        row = rows[index]
        results.append({
            "id": row["id"],
            "title": row["title"],
            "section": row["section"],
            "text": row["text"],
            "source_name": row["source_name"],
            "source_url": row["source_url"],
            "license": row["license"],
            "layer": row["layer"],
            "score": round(score, 3),
            "fetched_at": row["fetched_at"],
        })
    return results


def version() -> str:
    """知识库版本：片段数 + 最新抓取时间。重建索引后键自动变化。"""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(fetched_at), '') AS latest FROM chunks"
        ).fetchone()
    return f"{row['n']}@{row['latest']}" if row else "0@"
