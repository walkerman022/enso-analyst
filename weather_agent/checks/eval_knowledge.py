#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识问答评估（RAG 分支）。

检查四件事：
1. 知识类问题真的调了检索（回答里有 [kb:xx]）；
2. 引用编号不是编的（回答里不出现"引用不存在"的护栏提示）；
3. 关键概念有命中；
4. 知识库没有的问题，明确说没有，而不是编一段。

    python weather_agent/checks/eval_knowledge.py
    python weather_agent/checks/eval_knowledge.py --verbose
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

CITATION = re.compile(r"\[kb:[A-Za-z0-9._-]+\]")

CASES = [
    {
        "id": "def_oni",
        "question": "ONI 是怎么定义的？",
        "keywords": ["三个月", "滑动", "3.4"],
        "need_citation": True,
    },
    {
        "id": "ep_cp",
        "question": "东部型和中部型的区别是什么？",
        "keywords": ["东部型", "中部型"],
        "need_citation": True,
    },
    {
        "id": "easm",
        "question": "厄尔尼诺对东亚夏季风有什么影响？",
        "keywords": ["副热带高压", "长江"],
        "need_citation": True,
    },
    {
        "id": "five_season",
        "question": "五季判据是什么？",
        "keywords": ["连续", "5"],
        "need_citation": True,
    },
    {
        "id": "cross_lingual",
        "question": "为什么春季的 ENSO 预报最难？",
        "keywords": ["春季", "预报", "障碍"],
        "need_citation": True,
    },
    {
        "id": "out_of_scope",
        "question": "厄尔尼诺对南极海冰有什么影响？",
        "keywords": ["没有", "未", "资料", "知识库"],
        "need_citation": False,
    },
]


def run_case(agent, case: dict, verbose: bool) -> tuple[bool, list[str]]:
    evidence.reset()
    text = agent.run(case["question"])
    text = text if isinstance(text, str) else str(text)

    details: list[str] = []
    ok = True

    has_citation = bool(CITATION.search(text))
    if case["need_citation"]:
        ok = ok and has_citation
        details.append(f"{'✓' if has_citation else '✗'} 含知识库引用")

    fabricated = "引用不存在" in text
    ok = ok and not fabricated
    details.append(f"{'✓' if not fabricated else '✗'} 无编造引用")

    hit = any(keyword.lower() in text.lower() for keyword in case["keywords"])
    ok = ok and hit
    details.append(f"{'✓' if hit else '✗'} 命中关键词 {case['keywords']}")

    if verbose:
        print(f"\n--- {case['id']} ---\n{text}\n")
    return ok, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知识问答评估")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", help="只跑指定 id，逗号分隔")
    args = parser.parse_args(argv)

    cases = CASES
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]

    agent = build_agent()
    passed = 0
    for index, case in enumerate(cases, start=1):
        try:
            ok, details = run_case(agent, case, args.verbose)
        except Exception as exc:
            ok, details = False, [f"✗ 执行异常 {type(exc).__name__}: {exc}"]
        passed += int(ok)
        print(f"[{index}/{len(cases)}] {'PASS' if ok else 'FAIL'} {case['id']}: {case['question']}")
        if not ok:
            for line in details:
                if line.startswith("✗"):
                    print(f"    {line}")

    print(f"\n合计 {len(cases)} 条，通过 {passed}，失败 {len(cases) - passed}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
