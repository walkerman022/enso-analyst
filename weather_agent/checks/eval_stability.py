#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""稳定性评估：同一问题重复多次，看结论是否一致。

为什么单独做：单次通过不代表可靠。同一问题三次答案里数字不同、结论方向相反，
说明提示词有歧义或数据在变，这比"偶尔答错"更值得知道。

判定方式（两级）：
- 核心一致：三次回答共同出现的数值非空。只要核心数值一致，说明结论没飘。
- 整体重叠：两两 Jaccard 平均值，用来观察"外围细节"波动多大。
判 PASS 看核心一致；整体重叠只作参考。没有数值的回答（如概念题）改看方向词是否一致。

为什么不用完全相等：模型有时多列一个邻季值、多提一次历史对比，结论没变，
完全相等会把这些判成"不稳定"，指标就失去意义。

    python weather_agent/checks/eval_stability.py
    python weather_agent/checks/eval_stability.py --repeat 3 --verbose
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

from weather_agent import evidence, metrics  # noqa: E402
from weather_agent.agent import build_agent  # noqa: E402

NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+")
DIRECTIONS = {
    "el_nino": ("厄尔尼诺", "el niño", "el nino"),
    "la_nina": ("拉尼娜", "la niña", "la nina"),
    "neutral": ("中性", "未构成", "不足"),
    "uncovered": ("未覆盖", "没有发布", "尚未发布"),
    "rising": ("上升", "增强", "升高", "升温"),
    "falling": ("下降", "减弱", "降低", "降温"),
}

CASES = [
    {"q": "最近一期 ONI 是多少？"},
    {"q": "现在算不算厄尔尼诺？"},
    {"q": "1997 年那次厄尔尼诺峰值多少？"},
    {"q": "最近五季 ONI 在上升还是下降？"},
    # 概念题没有数值，改看关键内容是否每次都出现
    {"q": "厄尔尼诺对东亚夏季风有什么影响？", "expect": ["副热带高压", "长江", "夏季风"]},
]


def _numbers(text: str) -> set[str]:
    return {value for value in NUMBER.findall(text)}


def _directions(text: str) -> set[str]:
    lowered = text.lower()
    return {label for label, words in DIRECTIONS.items() if any(word in lowered for word in words)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="稳定性评估")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    stable = 0
    for index, case in enumerate(CASES, start=1):
        question = case["q"]
        runs: list[str] = []
        for _ in range(max(2, args.repeat)):
            evidence.reset()
            agent = build_agent()
            text = agent.run(question)
            runs.append(text if isinstance(text, str) else str(text))

        number_sets = [_numbers(item) for item in runs]
        directions = [_directions(item) for item in runs]

        pairs = [
            _jaccard(number_sets[i], number_sets[j])
            for i in range(len(number_sets)) for j in range(i + 1, len(number_sets))
        ]
        overlap = round(sum(pairs) / len(pairs), 2) if pairs else 1.0
        core = set.intersection(*number_sets) if number_sets and all(number_sets) else set()
        direction_agree = len({frozenset(item) for item in directions}) == 1
        if any(number_sets):
            ok = bool(core)
        else:
            expected = case.get("expect") or []
            ok = all(any(word in item for word in expected) for item in runs) if expected else direction_agree
        stable += int(ok)

        print(f"[{index}/{len(CASES)}] {'PASS' if ok else 'FAIL'} {question}"
              f"（核心数值 {len(core)} 个，整体重叠 {overlap}）")
        if not ok:
            print(f"        数字差异：{[sorted(s) for s in number_sets]}")
        if not direction_agree:
            print(f"        方向提示（仅供参考）：{['/'.join(sorted(d)) or 'unknown' for d in directions]}")
        if args.verbose and not ok:
            for i, text in enumerate(runs, start=1):
                print(f"        --- 第 {i} 次 ---\n{text[:300]}")

    print(f"\n一致 {stable}/{len(CASES)}，稳定率 {stable / len(CASES):.0%}（判据：核心数值三次都出现；概念题看关键词覆盖）")
    print(f"（每条重复 {max(2, args.repeat)} 次，共 {len(CASES) * max(2, args.repeat)} 次调用）")
    return 0 if stable == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
