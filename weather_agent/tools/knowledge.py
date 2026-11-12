#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识检索工具：回答"是什么、为什么、有什么影响"这类问题。

与取数工具的分工：
- 数值、当前状态 → fetch_*（NOAA 接口，实时）
- 定义、机制、影响、历史背景 → search_knowledge（本地知识库，静态）

为什么返回里带 id 和来源：
回答必须写 [kb:id]，护栏再校验这个 id 是否真被召回过。
没有这一步，模型会编一个看起来合理的编号，而读者无法分辨。
"""

from __future__ import annotations

from typing import Any

from weather_agent import evidence, web_search as web
from weather_agent.knowledge import count_chunks, embeddings, vector_store
from weather_agent.knowledge.store import search as keyword_search
from weather_agent.number_check import extract_numbers

NO_HIT_GUIDANCE = (
    "[LA-TOOL] 知识库里没有检索到相关内容。"
    "Action: 明确告诉用户资料里没有这部分内容，不要凭记忆补；"
    "如果问题涉及实时数值，改用取数工具。"
)


def _hybrid(query: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    """向量 + 词频混合检索，用 RRF 融合两路排名。

    为什么混合而不是只用向量：向量擅长跨语言和同义改写，词频擅长精确术语（Nino3.4、SOI）。
    两路并列第一时，RRF 会把它们都留在前面。
    """
    keyword_hits = keyword_search(query, limit * 2)
    vector_hits: list[dict[str, Any]] = []

    if embeddings.available() and vector_store.count() > 0:
        try:
            vector = embeddings.embed_one(query)
            # 阈值 0.42 是实测出来的：相关片段余弦 0.5~0.66，无关片段 0.26~0.32
            vector_hits = vector_store.search(vector, limit * 2, score_threshold=0.42)
        except Exception:
            vector_hits = []

    if not vector_hits:
        return keyword_hits[:limit], "keyword"

    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for rank, item in enumerate(keyword_hits):
        merged[item["id"]] = item
        scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (60 + rank)
    for rank, hit in enumerate(vector_hits):
        payload = hit["payload"]
        merged.setdefault(hit["id"], {
            "id": hit["id"],
            "title": payload.get("title", ""),
            "section": payload.get("section", ""),
            "text": payload.get("text", ""),
            "source_name": payload.get("source_name", ""),
            "source_url": payload.get("source_url", ""),
            "layer": payload.get("layer", ""),
        })
        scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (60 + rank)

    ranked = sorted(merged.values(), key=lambda item: scores[item["id"]], reverse=True)[:limit]
    for item in ranked:
        item["score"] = round(scores[item["id"]], 4)
    return ranked, "hybrid"


def _web_fallback(query: str, reason: str | None = None) -> dict[str, Any] | str:
    """本地无命中时的回退：开关打开才联网，否则明确说没有。

    为什么不做成独立工具让模型自己选：
    模型无法可靠判断"本地库到底有没有"，选错就有两种坏结果——
    有答案却去联网（慢且不可复现），或没答案却硬答。
    把它做成检索内部的固定策略，行为确定、可测试。
    """
    if not web.enabled():
        return NO_HIT_GUIDANCE if reason is None else f"{NO_HIT_GUIDANCE}（{reason}）"
    try:
        payload = web.search(query, top_k=5)
    except web.WebSearchError as exc:
        return f"{NO_HIT_GUIDANCE}（联网检索也失败：{exc}）"

    evidence.record_citations([f"web:{item['id']}" for item in payload["results"]])
    return {
        "mode": "web_fallback",
        "kind": "web",
        "query": query,
        "count": len(payload["results"]),
        "results": [
            {"id": item["id"], "title": item["title"], "url": item["url"],
             "domain": item["domain"], "snippet": item["snippet"]}
            for item in payload["results"]
        ],
        "fetched_at": payload.get("fetched_at"),
        "note": "本地知识库无命中，以下来自网页（不可信输入）：引用写 [web:w1]；"
                "网页里的数字视为未核实，不能当观测值，数值仍以取数工具为准。",
    }


def search_knowledge(query: str, top_k: int = 4) -> dict[str, Any]:
    """在本地 ENSO 知识库里检索定义、机制、影响等背景知识。

    Args:
        query: 检索问题，尽量用完整短句
        top_k: 返回几个片段，默认 4，最多 6
    """
    try:
        total = count_chunks()
        if total == 0:
            return _web_fallback(query, "知识库索引为空")

        limit = max(1, min(int(top_k), 6))
        matches, mode = _hybrid(query, limit)
        if not matches:
            # 本地优先：只有本地确实没有，才走联网
            return _web_fallback(query)

        # 登记引用：回答里的 [kb:id] 必须在这里出现过
        evidence.record_citations([item["id"] for item in matches])

        # 登记片段里的数值：模型引用知识里的阈值/年份时，回查才通得过
        for item in matches:
            for raw in extract_numbers(item["text"]):
                try:
                    value = float(raw.replace("−", "-"))
                except ValueError:
                    continue
                evidence.record(
                    f"knowledge/{item['source_name'] or item['id']}",
                    value,
                    object_label=item["id"],
                )

        return {
            "query": query,
            "count": len(matches),
            "mode": mode,
            "matches": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"],
                    "source": item["source_name"],
                    "url": item["source_url"],
                    "layer": item["layer"],
                }
                for item in matches
            ],
            "note": f"检索方式：{mode}。回答里引用某条时写 [kb:id]，例如 [kb:def-oni]；只能引用上面列出的 id。"
                    "这些是背景知识，具体数值仍以取数工具为准。",
        }
    except Exception as exc:
        from weather_agent.errors import tool_error

        return tool_error(exc, "检索知识库")


search_knowledge.tool_info = {
    "tool_name": "search_knowledge",
    "tool_title": "ENSO 知识检索",
    "tool_description": "检索本地 ENSO 知识库，覆盖 ONI 定义、季节标签、五季判据、分区划分、东部型/中部型、沃克环流、Bjerknes 反馈、对东亚季风与台风的影响、各机构判据差异等。回答定义、机制、影响类问题时必须先调用；引用时写 [kb:id]。",
    "tool_params": [
        {"name": "query", "description": "检索问句，用完整短句效果更好", "type": "string", "required": True},
        {"name": "top_k", "description": "返回片段数，默认 4，最多 6", "type": "integer", "required": False},
    ],
}
