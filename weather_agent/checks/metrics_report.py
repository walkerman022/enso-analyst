#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行度量报表：读 metrics.db，输出延迟、轮次、token、护栏触发率。

    python weather_agent/checks/metrics_report.py
    python weather_agent/checks/metrics_report.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行度量报表")
    parser.add_argument("--limit", type=int, default=200, help="取最近多少条记录")
    args = parser.parse_args(argv)

    data = metrics.summary(limit=args.limit)
    if data.get("count", 0) == 0:
        print("还没有运行记录。跑一次 CLI 问答就会写入。")
        return 0

    print(f"样本数：{data['count']}")
    print(f"成功率：{data['success_rate']:.1%}")
    print(f"延迟：P50 {data['latency_p50_ms']} ms / P95 {data['latency_p95_ms']} ms / 均值 {data['latency_avg_ms']} ms")
    print(f"平均模型调用：{data['model_calls_avg']} 次")
    print(f"平均工具调用：{data['tool_calls_avg']} 次")
    print(f"平均 token：{data['tokens_avg']}")
    print(f"护栏触发率：{data['guardrail_rate']:.1%}")
    print(f"含引用比例：{data['citation_rate']:.1%}")
    print(f"降级比例（无 token 计数）：{data['degraded_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
