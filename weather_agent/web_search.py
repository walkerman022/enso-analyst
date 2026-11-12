#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""联网搜索（默认关闭）。

为什么默认关闭：
- 网页内容不可复现，会破坏“引用可回查”的前提；
- 延迟比本地检索高一个量级（2–5 秒 vs 60–680 毫秒）；
- 网页是未信任输入，可能含指令性文本。

开启方式：`--web` 参数，或环境变量 WEB_SEARCH_ENABLED=1。

安全边界：
- 只请求固定的搜索域名，不抓取结果页（不跟随用户可控 URL，天然规避 SSRF）；
- 结果只保留标题、链接、摘要，正文不落库；
- 摘要统一截断，避免把长网页塞进上下文。
"""

from __future__ import annotations

import html as html_lib
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

SEARCH_URL = "https://cn.bing.com/search?q={query}&setlang=zh-Hans"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) weather-agent/0.1"
MAX_BYTES = 2 * 1024 * 1024
SNIPPET_CHARS = 180
BLOCK_RE = re.compile(r'<li class="b_algo".*?</li>', re.S)
TITLE_RE = re.compile(r"<h2[^>]*>.*?<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S)
SNIPPET_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
# 网页是未信任输入：把常见“指令注入”句式替换掉，而不是原样塞给模型。
INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions?"
    r"|disregard\s+(all\s+)?(previous|prior)\s+instructions?"
    r"|忽略(以上|之前|先前|前面)的?(所有)?(指令|规则|提示)"
    r"|你现在是|system\s*prompt|系统提示词)",
    re.I,
)


class WebSearchError(RuntimeError):
    """联网搜索不可用（关闭、网络、解析失败）。"""


def enabled() -> bool:
    return str(os.getenv("WEB_SEARCH_ENABLED", "")).strip().lower() in ("1", "true", "on", "yes")


def _clean(text: str) -> str:
    text = TAG_RE.sub("", text or "")
    text = html_lib.unescape(text)
    text = INJECTION_RE.sub("[已移除可疑指令]", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_html(query: str, timeout: float = 12.0) -> str:
    url = SEARCH_URL.format(query=urllib.parse.quote_plus(query))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_BYTES)
    except Exception as exc:
        raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc
    return raw.decode("utf-8", errors="replace")


def parse_results(html: str, top_k: int = 5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in BLOCK_RE.findall(html or ""):
        title_match = TITLE_RE.search(block)
        if not title_match:
            continue
        url, title = title_match.group(1), _clean(title_match.group(2))
        snippet_match = SNIPPET_RE.search(block)
        snippet = _clean(snippet_match.group(1)) if snippet_match else ""
        if not title or not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        domain = urllib.parse.urlparse(url).netloc
        results.append({
            "id": f"w{len(results) + 1}",
            "title": title,
            "url": url,
            "domain": domain,
            "snippet": snippet[:SNIPPET_CHARS],
        })
        if len(results) >= max(1, int(top_k)):
            break
    return results


def search(query: str, top_k: int = 5) -> dict[str, Any]:
    if not enabled():
        raise WebSearchError(
            "联网搜索未开启。Action: 用 --web 启动，或设置 WEB_SEARCH_ENABLED=1；"
            "未开启时请只用本地知识库，并明确告知用户资料范围。"
        )
    if not (query or "").strip():
        raise WebSearchError("查询为空")
    results = parse_results(_fetch_html(query), top_k=top_k)
    if not results:
        raise WebSearchError("搜索页没有解析到结果（页面结构可能变化）")
    return {
        "query": query,
        "results": results,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "bing-html",
    }
