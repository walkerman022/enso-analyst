#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会商模式评估（多 Agent）。每条 = 3 次专家调用 + 1 次首席调用。

检查四件事：
1. 三个角色都产出了报告，且各自账本非空（说明真的查了数据）；
2. 首席输出含固定小节（结论 / 一致点 / 分歧 / 置信度）；
3. 首席引用的数字都能在合并账本里找到（防止它自己编数）；
4. 报告里出现了应有关键词（例如"El Niño"、"样本"）。

    python weather_agent/checks/eval_council.py
    python weather_agent/checks/eval_council.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import number_check  # noqa: E402
from weather_agent.council import EXPERT_KEYS, format_minutes, run_council  # noqa: E402

SECTIONS = ("结论", "一致点", "分歧", "置信度")

CASES = [
    {
        "id": "state",
        "question": "现在厄尔尼诺到什么程度了？",
        "keywords": ["厄尔尼诺", "el niño", "el nino"],
        "require_sample_note": False,
    },
    {
        "id": "outlook",
        "question": "这次事件接下来可能会怎么演变？",
        "keywords": ["样本", "历史", "季节"],
        "require_sample_note": True,
    },
]


def run_case(case: dict, verbose: bool) -> tuple[bool, list[str]]:
    result = run_council(case["question"])
    details: list[str] = []
    ok = True

    for item in result["reports"]:
        if not item["report"].strip():
            ok = False
            details.append(f"✗ {item['key']} 报告为空")
        if item["evidence_count"] == 0:
            ok = False
            details.append(f"✗ {item['key']} 没查到任何数据（账本为空）")

    verdict = result["verdict"]
    for section in SECTIONS:
        passed = section in verdict
        ok = ok and passed
        details.append(f"{'✓' if passed else '✗'} 首席含小节「{section}」")

    check = number_check.verify(verdict, scope="council")
    passed = check.ok
    ok = ok and passed
    details.append(f"{'✓' if passed else '✗'} 首席数值可追溯（{check.summary()}）")

    text = verdict + "".join(item["report"] for item in result["reports"])
    passed = any(word.lower() in text.lower() for word in case["keywords"])
    ok = ok and passed
    details.append(f"{'✓' if passed else '✗'} 命中关键词 {case['keywords']}")

    if case["require_sample_note"]:
        passed = "样本" in text
        ok = ok and passed
        details.append(f"{'✓' if passed else '✗'} 提到历史样本量")

    passed = "各组原始意见" in format_minutes(result)
    ok = ok and passed
    details.append(f"{'✓' if passed else '✗'} 纪要含各组原始意见")

    if verbose:
        print("\n" + format_minutes(result) + "\n")
    return ok, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="会商模式评估")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    passed = 0
    for index, case in enumerate(CASES, start=1):
        try:
            ok, details = run_case(case, args.verbose)
        except Exception as exc:
            ok, details = False, [f"✗ 执行异常 {type(exc).__name__}: {exc}"]
        passed += int(ok)
        print(f"[{index}/{len(CASES)}] {'PASS' if ok else 'FAIL'} {case['id']}: {case['question']}")
        for line in details:
            if not ok or args.verbose:
                print(f"    {line}")

    print(f"\n合计 {len(CASES)} 条，通过 {passed}，失败 {len(CASES) - passed}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
