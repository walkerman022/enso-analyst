#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测能力评估：检查 Agent 会不会正确区分"预测"和"观测"。

两条用例：
1. 预测类问题 → 必须调用 predict_enso，回答标注"预测/预计"、模型与数据截止；不得说成当前观测；
2. 当前状态问题 → 必须走取数工具，不得用预测值回答"现在"。

    python weather_agent/checks/eval_prediction.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import evidence  # noqa: E402
from weather_agent.agent import build_agent  # noqa: E402
from weather_agent.tool_recorder import ToolRecorder  # noqa: E402
from weather_agent.tools import ALL_TOOLS  # noqa: E402

CASES = [
    {
        "id": "forecast",
        "question": "预计未来三个月 Niño3.4 会怎么变化？",
        "must_call": "predict_enso",
        "forbid_call": None,
        "must_match": ["预测|预计|模型"],
        "forbid_match": ["当前 ONI 为|目前 ONI 为"],
    },
    {
        "id": "current_status",
        "question": "现在厄尔尼诺到什么程度了？",
        "must_call": None,
        "forbid_call": "predict_enso",
        "must_match": ["厄尔尼诺|El Niño|中性"],
        "forbid_match": [],
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预测能力评估")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", help="只跑指定 id")
    args = parser.parse_args(argv)

    recorder = ToolRecorder()
    agent = build_agent(tools=recorder.wrap(ALL_TOOLS))

    passed = 0
    cases = [c for c in CASES if not args.only or c['id'] == args.only]
    for index, case in enumerate(cases, start=1):
        recorder.reset()
        evidence.reset()
        text = agent.run(case["question"])
        text = text if isinstance(text, str) else str(text)
        names = recorder.names()

        details = []
        ok = True
        if case["must_call"]:
            hit = case["must_call"] in names
            ok = ok and hit
            details.append(f"{'✓' if hit else '✗'} 调用了 {case['must_call']}：{names}")
        if case["forbid_call"]:
            missed = case["forbid_call"] not in names
            ok = ok and missed
            details.append(f"{'✓' if missed else '✗'} 未误用 {case['forbid_call']}：{names}")

        pattern = "|".join(case["must_match"])
        matched = bool(re.search(pattern, text))
        ok = ok and matched
        details.append(f"{'✓' if matched else '✗'} 命中 {pattern}")

        if case["forbid_match"]:
            banned = re.search("|".join(case["forbid_match"]), text)
            ok = ok and not banned
            details.append(f"{'✓' if not banned else '✗'} 未把预测说成当前（{banned.group(0) if banned else '无'}）")

        passed += int(ok)
        print(f"[{index}/{len(CASES)}] {'PASS' if ok else 'FAIL'} {case['id']}: {case['question']}")
        if not ok or args.verbose:
            for line in details:
                print(f"    {line}")
            if args.verbose:
                print(f"\n{text[:400]}\n")

    print(f"\n合计 {len(cases)} 条，通过 {passed}，失败 {len(cases) - passed}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
