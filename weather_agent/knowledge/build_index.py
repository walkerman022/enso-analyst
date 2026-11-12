#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建知识库索引：抓官方页面 + 写自编卡片，切成片段存进 SQLite。

三层语料：
1. 官方文档（NOAA Climate.gov / CPC / NCEI、国家气候中心）——只摘正文段落，保留来源链接
2. 综述/FAQ（CPC ENSO FAQ）——作为百科层的替代（维基百科在本机网络不可达）
3. 自编卡片（cards.py）——定义、判据、机制、影响

用法：
    python -m weather_agent.knowledge.build_index            # 用缓存，缺了才下载
    python -m weather_agent.knowledge.build_index --refresh  # 强制重新抓
    python -m weather_agent.knowledge.build_index --cards-only  # 只写卡片（离线/测试用）
"""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from typing import Any

from weather_agent.cache import DataSourceError, fetch_text
from weather_agent.knowledge.cards import CARDS
from weather_agent.knowledge import embeddings, vector_store
from weather_agent.knowledge.store import add_chunks, count_chunks, init_db

# 抓取目标。license 只写"公共领域/摘录引用"，具体以原站声明为准。
SOURCES: list[dict[str, str]] = [
    {
        "doc_id": "noaa-climate-gov",
        "title": "NOAA Climate.gov · ENSO",
        "url": "https://www.climate.gov/enso",
        "source_name": "NOAA Climate.gov",
        "license": "U.S. Government work (public domain)",
        "layer": "official",
    },
    {
        "doc_id": "cpc-advisory",
        "title": "NOAA CPC · ENSO Diagnostic Discussion",
        "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml",
        "source_name": "NOAA CPC",
        "license": "U.S. Government work (public domain)",
        "layer": "official",
    },
    {
        "doc_id": "ncei-enso",
        "title": "NOAA NCEI · ENSO monitoring",
        "url": "https://www.ncei.noaa.gov/access/monitoring/enso/",
        "source_name": "NOAA NCEI",
        "license": "U.S. Government work (public domain)",
        "layer": "official",
    },
    {
        "doc_id": "cpc-faq",
        "title": "NOAA CPC · ENSO FAQ",
        "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ensofaq.shtml",
        "source_name": "NOAA CPC",
        "license": "U.S. Government work (public domain)",
        "layer": "reference",
    },
    {
        "doc_id": "ncc-cma",
        "title": "国家气候中心 · ENSO 监测",
        "url": "https://www.ncc-cma.net/",
        "source_name": "国家气候中心",
        "license": "摘录引用（原文版权归原站）",
        "layer": "official",
    },
]

_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "dd", "dt"}


class _TextExtractor(HTMLParser):
    """把 HTML 拆成 (标签, 文本) 块，只收正文类标签。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._tag: str | None = None
        self._buffer: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth == 0 and self._tag:
            text = data.strip()
            if text:
                self._buffer.append(text)

    def _flush(self) -> None:
        if self._tag and self._buffer:
            text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
            if len(text) >= 20:
                self.blocks.append((self._tag, text))
        self._tag = None
        self._buffer = []

    def close(self) -> None:
        self._flush()
        super().close()


def extract_blocks(html: str) -> list[tuple[str, str]]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.blocks


_SENTENCE_END = "。！？!?；;\n"


def _split_long(text: str, max_chars: int) -> list[str]:
    """单段超过上限时按句切；句子仍超长再硬切。

    为什么需要：原先一段 2812 字符会独占一片，embedding 模型只取前 512 token，
    后半段等于没进检索。这里只对"超长单段"生效，不影响正常段落。
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in _SENTENCE_END and len(buffer) >= max_chars * 0.6:
            pieces.append(buffer)
            buffer = ""
    if buffer:
        pieces.append(buffer)

    result: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            result.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            result.append(piece)
    return result


def chunk_blocks(blocks: list[tuple[str, str]], max_chars: int = 700) -> list[tuple[str, str]]:
    """按标题分组，长段落累积到 max_chars 就切一片。返回 (section, text)。"""
    chunks: list[tuple[str, str]] = []
    section = ""
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            text = " ".join(buffer)
            for piece in _split_long(text, max_chars):
                chunks.append((section, piece))
        buffer = []
        size = 0

    for tag, text in blocks:
        if tag in {"h1", "h2", "h3", "h4"}:
            flush()
            section = text[:60]
            continue
        if size + len(text) > max_chars:
            flush()
        buffer.append(text)
        size += len(text)
    flush()
    return chunks


def build(refresh: bool = False, cards_only: bool = False, verbose: bool = True) -> dict[str, Any]:
    init_db(reset=True)
    stats: dict[str, Any] = {"cards": 0, "fetched": 0, "failed": [], "vectors": 0}
    all_chunks: list[dict[str, Any]] = []

    cards = [
        {
            **card,
            "doc_id": f"card-{card['id']}",
            "section": "自编卡片",
            "license": "自编摘要（依据公开资料）",
            "layer": "card",
        }
        for card in CARDS
    ]
    stats["cards"] = add_chunks(cards)
    all_chunks.extend(cards)
    if verbose:
        print(f"卡片写入 {stats['cards']} 条")

    if cards_only:
        stats["total"] = count_chunks()
        return stats

    for source in SOURCES:
        try:
            result = fetch_text(source["url"], force=refresh)
            blocks = extract_blocks(result["text"])
            pieces = chunk_blocks(blocks)
            payload = [
                {
                    "id": f"{source['doc_id']}-{index:03d}",
                    "doc_id": source["doc_id"],
                    "title": source["title"],
                    "section": section,
                    "text": text,
                    "source_url": source["url"],
                    "source_name": source["source_name"],
                    "license": source["license"],
                    "layer": source["layer"],
                    "fetched_at": result.get("updated_at"),
                }
                for index, (section, text) in enumerate(pieces)
            ]
            stats["fetched"] += add_chunks(payload)
            all_chunks.extend(payload)
            if verbose:
                print(f"{source['doc_id']}: {len(payload)} 片")
        except (DataSourceError, Exception) as exc:  # 单个源失败不影响整体
            stats["failed"].append({"doc_id": source["doc_id"], "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"{source['doc_id']}: 失败 {type(exc).__name__}")

    stats["total"] = count_chunks()
    stats["vectors"] = _build_vectors(all_chunks, verbose=verbose)

    # 知识库变了，缓存里引用旧片段的答案必须失效
    try:
        from weather_agent import prompt_cache

        prompt_cache.invalidate("kb")
    except Exception:
        pass
    return stats


def _build_vectors(chunks: list[dict[str, Any]], verbose: bool = True) -> int:
    """把片段写进 Qdrant。向量化失败不影响词频检索可用。"""
    if not chunks:
        return 0
    if not embeddings.available():
        if verbose:
            print("未配置 EMBEDDING_*，跳过向量索引（词频检索仍可用）")
        return 0
    try:
        texts = [f"{item.get('title', '')} {item.get('section', '')} {item['text']}" for item in chunks]
        vectors = embeddings.embed_texts(texts)
        vector_store.ensure_collection(len(vectors[0]), reset=True)
        points = [
            {
                "id": item["id"],
                "vector": vector,
                "payload": {
                    "title": item.get("title", ""),
                    "text": item["text"],
                    "source_name": item.get("source_name", ""),
                    "source_url": item.get("source_url", ""),
                    "layer": item.get("layer", ""),
                    "section": item.get("section", ""),
                },
            }
            for item, vector in zip(chunks, vectors)
        ]
        written = vector_store.upsert(points)
        if verbose:
            print(f"向量入库 {written} 条（维度 {len(vectors[0])}）")
        return written
    except Exception as exc:
        if verbose:
            print(f"向量索引失败，已降级为纯词频检索：{type(exc).__name__}: {exc}")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 ENSO 知识库索引")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新抓取")
    parser.add_argument("--cards-only", action="store_true", help="只写自编卡片")
    args = parser.parse_args(argv)

    stats = build(refresh=args.refresh, cards_only=args.cards_only)
    print(f"\n合计 {stats['total']} 片（卡片 {stats['cards']}，抓取 {stats['fetched']}），向量 {stats['vectors']} 条")
    if stats["failed"]:
        print("失败源：")
        for item in stats["failed"]:
            print(f"  - {item['doc_id']}: {item['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
