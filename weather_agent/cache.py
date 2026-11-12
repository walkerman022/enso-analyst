#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""带 TTL 的文本下载缓存。

为什么要自己做一层缓存：
1. 同一次会话里模型可能连着问三次"最近 ONI 多少"，每次都去敲 NOAA 既慢又没礼貌；
2. NOAA 偶尔抽风/本机网络抖动，缓存能让"刚才还能查的数据"继续可用；
3. 跑测试时不希望每次都打网络（可以把 ttl 设很大或用 force=False 复用）。

缓存键取 URL 的 sha1，元数据单独存一个 json，正文存 txt。
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# 默认 6 小时。ONI 是月/季尺度数据，一天更新不了几次，6 小时足够新。
DEFAULT_TTL = 6 * 3600

# NOAA 对没有 UA 的请求有时候会 403，带上一个能识别身份的 UA。
USER_AGENT = "weather-agent/0.1 (ENSO research; local use)"


class DataSourceError(RuntimeError):
    """数据源不可用（网络、HTTP 状态码、超时）。

    单独定义一类异常，是为了让工具层能区分
    "数据源挂了"（要告诉用户查不到）和 "解析失败"（要报 bug）。
    """


def _cache_paths(url: str) -> tuple[Path, Path]:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{key}.txt", CACHE_DIR / f"{key}.json"


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 元数据坏了就当没有缓存，重新下载；不要因为缓存损坏让整个工具挂掉。
        return None


def fetch_text(
        url: str,
        *,
        ttl: float = DEFAULT_TTL,
        timeout: float = 20.0,
        force: bool = False,
) -> dict[str, Any]:
    """下载文本，命中缓存则直接返回。

    返回：
        {
          "text": 正文,
          "updated_at": 数据源最后修改时间（HTTP Last-Modified，拿不到就是 None）,
          "fetched_at": 本地抓取时间（epoch 秒）,
          "from_cache": 是否来自缓存,
          "url": 原始 URL,
        }

    失败抛 DataSourceError，由调用方决定怎么呈现给模型。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text_path, meta_path = _cache_paths(url)

    if not force and text_path.exists():
        meta = _read_meta(meta_path)
        if meta and (time.time() - float(meta.get("fetched_at", 0))) < ttl:
            return {
                "text": text_path.read_text(encoding="utf-8"),
                "updated_at": meta.get("updated_at"),
                "fetched_at": meta.get("fetched_at"),
                "from_cache": True,
                "url": url,
            }

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            # NOAA 的 ascii 文件是纯 ASCII；用 replace 兜底，避免个别脏字符直接炸掉解析。
            text = raw.decode("utf-8", errors="replace")
            last_modified = response.headers.get("Last-Modified")
    except urllib.error.HTTPError as exc:
        raise DataSourceError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise DataSourceError(f"network error for {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DataSourceError(f"timeout after {timeout}s for {url}") from exc

    fetched_at = time.time()
    # 先写临时文件再 replace：避免进程中途被杀留下半个文件，下次读到残缺数据。
    tmp_path = text_path.with_suffix(".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(text_path)
    meta_path.write_text(
        json.dumps(
            {"url": url, "fetched_at": fetched_at, "updated_at": last_modified},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "text": text,
        "updated_at": last_modified,
        "fetched_at": fetched_at,
        "from_cache": False,
        "url": url,
    }
