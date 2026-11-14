#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成忠实度评分：被引用的陈述句，是否真的被所引片段支持。

定义（避免把"没引用的话"混进分母）：
- 只评"带 [kb:id] 的陈述句"：这些句子是模型明确声明有出处的
- 支持判据：句子与所引片段的词面重合 ≥ 阈值，或句中的数字出现在片段中
- 未引用的陈述句单独统计为"引用覆盖率"，不参与忠实度

为什么不用第二个模型当裁判：不可复现、没法进门禁。
为什么不用"与全部检索片段重合"：模型引用哪片就该由哪片支持，
用全部片段算会放水（引 A 却由 B 支持也算过）。

局限：同义改写可能被误判为不支持（"海温升高" vs "东太平洋增暖"），
所以它适合做退化监控，不适合当绝对结论。
"""

from __future__ import annotations

import re
from typing import Any

from weather_agent.knowledge.store import tokenize

SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CITATION_RE = re.compile(r"\[kb:([A-Za-z0-9._-]+)\]")
MIN_TOKENS = 5
SUPPORT_THRESHOLD = 0.35
META_PATTERNS = (
    "如果需要", "需要提醒", "我可以", "如果你需要", "以上", "结论**", "小结", "说明：",
)


def split_claims(answer: str) -> list[str]:
    """切成候选陈述句；去掉问句、过短句、纯列表符号和元话语。

    先做一次引用归位：模型常写成“……。[kb:def-oni]”，
    引用在句号之后，直接切句会把它甩到下一段，导致该句看起来没有出处。
    """
    normalized = re.sub(r"([。！？!?；;])\s*((?:\[kb:[^\]]+\])+)", r"\2\1", answer or "")
    claims: list[str] = []
    for raw in SENTENCE_RE.findall(normalized):
        text = raw.strip().lstrip("-*•· ").strip()
        if len(text) < 8 or text.endswith(("？", "?")):
            continue
        if text.startswith("|") or set(text) <= set("-|: "):
            continue
        if any(text.startswith(pattern) or pattern in text[:12] for pattern in META_PATTERNS):
            continue
        if len(tokenize(text)) < MIN_TOKENS:
            continue
        claims.append(text)
    return claims


def support_ratio(claim: str, context_tokens: set[str]) -> float:
    tokens = set(tokenize(CITATION_RE.sub(" ", claim)))
    if not tokens:
        return 0.0
    return len(tokens & context_tokens) / len(tokens)


def _tokens(texts: list[str]) -> tuple[set[str], set[str]]:
    tokens: set[str] = set()
    numbers: set[str] = set()
    for text in texts:
        tokens |= set(tokenize(text))
        numbers |= set(NUMBER_RE.findall(text))
    return tokens, numbers


def key_term_hits(claim: str, context_tokens: set[str], idf: dict[str, float]) -> tuple[int, list[str]]:
    """命中的高 IDF 词数。同义改写常换掉虚词，但会保留核心术语。"""
    tokens = {token for token in tokenize(CITATION_RE.sub(" ", claim)) if token in idf}
    ranked = sorted(tokens, key=lambda token: (-idf[token], token))[:3]
    hits = [token for token in ranked if token in context_tokens]
    return len(hits), hits


def score_answer(answer: str, chunks: dict[str, str], threshold: float = SUPPORT_THRESHOLD,
                 idf: dict[str, float] | None = None) -> dict[str, Any]:
    """chunks: 本轮检索/引用的片段 id -> 正文；idf 由评估脚本从整个语料算好传入。"""
    all_tokens, all_numbers = _tokens(list(chunks.values()))
    idf = idf or {}

    rows: list[dict[str, Any]] = []
    for claim in split_claims(answer):
        cited = CITATION_RE.findall(claim)
        texts = [chunks[cid] for cid in cited if cid in chunks]
        context_tokens, context_numbers = _tokens(texts) if texts else (all_tokens, all_numbers)
        ratio = support_ratio(claim, context_tokens)
        numbers = set(NUMBER_RE.findall(claim))
        number_hit = bool(numbers & context_numbers)
        hits, hit_terms = key_term_hits(claim, context_tokens, idf) if idf else (0, [])
        supported = ratio >= threshold or number_hit or hits >= 2
        rows.append({
            "claim": claim,
            "cited": cited,
            "overlap": round(ratio, 3),
            "numbers": sorted(numbers),
            "number_hit": number_hit,
            "key_hits": hit_terms,
            "supported": supported,
        })

    cited_rows = [row for row in rows if row["cited"]]
    supported = [row for row in cited_rows if row["supported"]]
    unsupported = [row for row in cited_rows if not row["supported"]]
    return {
        "claims": len(rows),
        "cited_claims": len(cited_rows),
        "cited_supported": len(supported),
        "faithfulness": round(len(supported) / len(cited_rows), 3) if cited_rows else 0.0,
        "citation_coverage": round(len(cited_rows) / len(rows), 3) if rows else 0.0,
        "unsupported": [row["claim"] for row in unsupported],
        "uncited": [row["claim"] for row in rows if not row["cited"]],
        "rows": rows,
    }
