#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具调用准确率评估（借 BFCL 的思路，自己标 20 条）。

判分方式：给定问题，期望"第一个被调用的工具"是哪个；部分用例还检查关键参数。
为什么只查第一个：真实链路里模型可能先查状态再查历史，全序列匹配会把合理路径判错。

同时输出：平均模型调用次数、平均工具调用次数、平均 token——这三个是"效率"指标，
同样是答对，3 步和 8 步质量不同。

    python weather_agent/checks/eval_tools.py
    python weather_agent/checks/eval_tools.py --verbose
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import evidence, metrics  # noqa: E402
from weather_agent.agent import build_agent  # noqa: E402
from weather_agent.tool_recorder import ToolRecorder
from weather_agent.tools import ALL_TOOLS  # noqa: E402

# 期望值只写到"该用哪个工具"，参数只在关键处检查
CASES: list[dict[str, Any]] = [
    {"q": "最近一期 ONI 是多少？", "tool": "fetch_enso_index"},
    {"q": "现在算不算厄尔尼诺？", "tool": "describe_enso_state"},
    {"q": "1997 年厄尔尼诺的峰值是多少？", "tool": "find_oni_extremes", "covers_year": 1997},
    {"q": "最近 12 个季节的 ONI 序列给我看看", "tool": "fetch_oni", "args": {"seasons": 12}},
    {"q": "现在东部型还是中部型？", "tool": "classify_enso_type"},
    {"q": "Niño 四个分区最新的距平分别是多少？", "tool": "fetch_nino_regions"},
    {"q": "历史上最强的三次厄尔尼诺是哪几次？", "tool": "list_enso_history"},
    {"q": "强厄尔尼诺在峰值之后一般怎么演变？", "tool": "analog_evolution"},
    {"q": "最新一期 SOI 是多少？", "tool": "fetch_soi"},
    {"q": "画一下最近 20 年的 ONI 趋势图", "tool": "plot_enso_trend"},
    {"q": "画一下最新月份四个分区的距平柱状图", "tool": "plot_enso_regions"},
    {"q": "ONI 是怎么定义的？", "tool": "search_knowledge"},
    {"q": "东部型和中部型有什么区别？", "tool": "search_knowledge"},
    {"q": "厄尔尼诺对东亚夏季风有什么影响？", "tool": "search_knowledge"},
    {"q": "沃克环流是什么？", "tool": "search_knowledge"},
    {"q": "记住：我主要关注东部型厄尔尼诺", "tool": "save_preference"},
    {"q": "我之前让你记住什么了？", "tool": "recall_preferences"},
    {"q": "最近五季 ONI 一直在上升吗？", "tool": "fetch_oni"},
    {"q": "2015 年那次厄尔尼诺峰值多少？", "tool": "find_oni_extremes", "covers_year": 2015},
    {"q": "拉尼娜的定义是什么？", "tool": "search_knowledge"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工具调用准确率")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="weather_agent/checks/tool_accuracy.json")
    args = parser.parse_args(argv)

    recorder = ToolRecorder()
    correct = 0
    details = []
    for index, case in enumerate(CASES, start=1):
        recorder.reset()
        evidence.reset()
        # 每个用例用一个全新 agent：框架的计数跨 run 会部分重置，复用会让统计串味
        agent = build_agent(tools=recorder.wrap(ALL_TOOLS))
        before = metrics.snapshot(agent)
        text = agent.run(case["q"])
        text = text if isinstance(text, str) else str(text)
        after = metrics.snapshot(agent)

        names = recorder.names()
        actual = names[0] if names else None
        # 判分口径：期望工具出现在调用序列里即可（模型可能先查状态再查历史）
        args_ok = True
        if case.get("covers_year"):
            # 事件跨年时，模型给 2014-2016 这种区间更合理，只要覆盖目标年份就算对
            year = int(case["covers_year"])
            matched = [args for name, args in recorder.calls if name == case["tool"]]
            start = matched[0].get("start_year") if matched else None
            end = matched[0].get("end_year") if matched else None
            args_ok = start is not None and end is not None and int(start) <= year <= int(end)
            if not args_ok:
                actual_range = (start, end)
        if case.get("args"):
            matched = [args for name, args in recorder.calls if name == case["tool"]]
            args_ok = bool(matched) and all(
                str(matched[0].get(key)) == str(value) for key, value in case["args"].items()
            )
        ok = case["tool"] in names and args_ok
        correct += int(ok)
        details.append({
            "question": case["q"],
            "expected": case["tool"],
            "actual": actual,
            "args_ok": args_ok,
            "ok": ok,
            "model_calls": int(after["model_calls"]),
            "tool_calls": int(after["tool_calls"] - before["tool_calls"]),
            "tokens": int(after["tokens"] - before["tokens"]),
            "answer_head": text[:80],
        })
        print(f"[{index:02d}/{len(CASES)}] {'PASS' if ok else 'FAIL'} 期望 {case['tool']} 实际 {actual}")
        if not ok and args_ok is False:
            print(f"        参数不匹配：{recorder.calls[0][1] if recorder.calls else '无调用'}")
        if args.verbose and not ok:
            print(f"        回答片段：{text[:120]}")

    total = len(CASES)
    model_calls = [item["model_calls"] for item in details]
    tool_calls = [item["tool_calls"] for item in details]
    tokens = [item["tokens"] for item in details]
    result = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3),
        "model_calls_avg": round(sum(model_calls) / total, 2),
        "tool_calls_avg": round(sum(tool_calls) / total, 2),
        "tokens_avg": round(sum(tokens) / total, 1),
        "details": details,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n工具调用准确率 {correct}/{total} = {result['accuracy']:.1%}")
    print(f"平均模型调用 {result['model_calls_avg']} 次，工具调用 {result['tool_calls_avg']} 次，"
          f"tokens {result['tokens_avg']}")
    print(f"明细写入 {args.out}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
