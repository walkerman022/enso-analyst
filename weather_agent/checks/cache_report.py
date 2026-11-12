#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提示缓存命中率报表。

    python weather_agent/checks/cache_report.py
    python weather_agent/checks/cache_report.py --clear-expired
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import prompt_cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提示缓存报表")
    parser.add_argument("--clear-expired", action="store_true", help="清理过期条目")
    args = parser.parse_args(argv)

    if args.clear_expired:
        removed = prompt_cache.cleanup_expired()
        print(f"清理过期条目 {removed} 条")

    data = prompt_cache.stats()
    total = data["hits"] + data["misses"]
    print(f"条目数：{data['entries']}")
    print(f"命中 {data['hits']} / 未命中 {data['misses']} / 合计 {total}")
    print(f"命中率：{data['hit_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
