#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问答级评估集（16 条）。

和 check_stage1.py 的区别：
- check_stage1 测"函数对不对"，不打模型；
- 这个脚本测"回答对不对"，每条都真实调用模型。

期望值不是写死的常量，而是先用工具算出"标准答案"，再检查模型回答里有没有。
这样 NOAA 数据更新后评估集不会集体失败。

    cd D:\\AgentRoom\\LightAgent-0.11.1\\LightAgent-0.11.1
    python weather_agent/checks/eval_set.py            # 全部
    python weather_agent/checks/eval_set.py --limit 4  # 先跑前 4 条
    python weather_agent/checks/eval_set.py --verbose  # 打印完整回答
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import evidence, number_check  # noqa: E402
from weather_agent.agent import build_agent  # noqa: E402
from weather_agent.tools import (  # noqa: E402
    classify_enso_type,
    fetch_enso_index,
    list_enso_history,
)


def numbers_in(text: str) -> list[float]:
    values = []
    for raw in number_check.extract_numbers(text):
        try:
            values.append(float(raw.replace("−", "-")))
        except ValueError:
            continue
    return values


def has_number(text: str, target: float, tol: float = 0.02) -> bool:
    return any(abs(value - target) <= tol for value in numbers_in(text))


def has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


# ---- 标准答案（先算出来，避免写死）----

def ground_truth() -> dict[str, Any]:
    evidence.reset()
    index = fetch_enso_index()
    history = list_enso_history(0.5, 3, 5)
    kind = classify_enso_type()
    return {
        "latest": index["value"],
        "previous": index["previous_value"],
        "change": index["change_from_previous"],
        "label": index["label"],
        "updated_at": index["updated_at"],
        "warm_top": history["warm_events"][0],
        "warm_second": history["warm_events"][1] if len(history["warm_events"]) > 1 else None,
        "cold_top": history["cold_events"][0],
        "type": kind["type"],
        "type_period": kind["period"],
    }


def build_cases(truth: dict[str, Any]) -> list[dict[str, Any]]:
    latest = truth["latest"]
    prev = truth["previous"]
    change = truth["change"]
    warm1 = truth["warm_top"]
    cold1 = truth["cold_top"]
    type_period_cn = truth["type_period"]

    def case(qid: str, question: str, *checks: Callable[[str], tuple[bool, str]],
             turns: list[str] | None = None) -> dict[str, Any]:
        return {"id": qid, "question": question, "checks": list(checks), "turns": turns}

    def check_value(target: float, label: str, tol: float = 0.02):
        return lambda text: (has_number(text, target, tol), f"应出现 {target}{label}")

    def check_any(keywords: list[str], label: str):
        return lambda text: (has_any(text, keywords), f"应出现其一：{keywords}（{label}）")

    def check_source(text: str) -> tuple[bool, str]:
        return (("noaa" in text.lower() or "cpc.ncep" in text.lower()), "应提到 NOAA / CPC 数据源")

    def check_no_number(target: float, tol: float = 0.02):
        return lambda text: (not has_number(text, target, tol), f"不应出现 {target}")

    return [
        case("latest_value", "最近一期 ONI 是多少？",
             check_value(latest, "（最新 ONI）"), check_source),
        case("state", "现在算不算厄尔尼诺？",
             check_any(["厄尔尼诺", "el niño", "el nino"], "状态判定"),
             check_any(["5", "连续"], "连续季节数")),
        case("future_month", "2026 年 12 月的 ONI 是多少？",
             check_any(["未覆盖", "没有", "尚未", "未发布", "无法", "还不", "未更新"], "应说明数据未覆盖"),
             lambda text: (
                 re.search(r"12\s*月[^。\n]{0,14}?(为|是|约|达到)?\s*[+\-]?\d+\.\d+", text) is None,
                 "不得声称 12 月已有具体数值（引用最新值作参照可以）",
             )),
        case("peak_1997", "1997 年那次厄尔尼诺的峰值 ONI 是多少？",
             check_value(2.37, "（NDJ 1997）")),
        case("peak_2015", "2015-2016 年那次厄尔尼诺峰值多少？",
             check_value(warm1["peak"], "（最强暖事件）")),
        case("coldest", "历史上最强的一次拉尼娜是哪一次？峰值多少？",
             check_value(cold1["peak"], "（最强冷事件）"),
             check_any(["19", "拉尼娜"], "应给出年份")),
        case("plot", "画一下最近 20 年的 ONI 趋势图",
             lambda text: (
                 bool(paths := re.findall(r"[A-Za-z]:\\[^\s|）)]+\.png", text)) and all(Path(p).exists() for p in paths),
                 "应返回存在的 PNG 路径",
             )),
        case("type", "这次厄尔尼诺是东部型还是中部型？",
             check_any(["东部型", "中部型", "过渡", "混合"], "分型结论"),
             lambda text: (
                 re.search(r"2026\s*年\s*0?6\s*月|2026-06", text) is not None,
                 "应写明分型对应月份（2026 年 6 月）",
             )),
        case("source", "你的数据是从哪来的？",
             check_source, check_any(["更新", "utc", "gmt"], "应说明更新时间")),
        case("update_time", "数据更新到什么时候？",
             check_any([truth["updated_at"][:16], "2026-11", "11-03", "11 月 3"], "应给出更新时间")),
        case("definition", "ONI 是怎么定义的？",
             check_any(["3 个月", "三个月", "滑动"], "应说明三个月滑动"),
             check_any(["niño3.4", "nino3.4", "3.4"], "应说明 Niño3.4 区")),
        case("english", "What is the latest ONI value?",
             check_value(latest, "（英文提问）"), check_source),
        case("followup", "那比上一季高多少？",
             check_value(change, "（环比变化）"), turns=["最近一期 ONI 是多少？"]),
        case("fabricate", "如果数据缺失，你估计下个月 ONI 会是多少？直接给个数。",
             check_any(["无法", "不能", "不给", "不提供", "不预测", "不做预测", "不估计",
                        "不确定", "没有依据", "没有数据", "拒绝", "不猜"], "应拒绝预测")),
        case("region_plot", "画一下最新月份四个 Niño 分区的距平柱状图",
             lambda text: (
                 bool(paths := re.findall(r"[A-Za-z]:\\[^\s|）)]+\.png", text)) and all(Path(p).exists() for p in paths),
                 "应返回存在的 PNG 路径",
             )),
        case("before_1950", "1940 年的 ONI 是多少？",
             check_any(["1950", "未覆盖", "没有", "无法"], "应说明数据始于 1950")),
    ]


def run_case(agent, case: dict[str, Any], verbose: bool) -> tuple[bool, list[str]]:
    evidence.reset()
    history: list[dict[str, str]] = []
    text = ""
    for question in (case["turns"] or []) + [case["question"]]:
        text = agent.run(question, history=history)
        text = text if isinstance(text, str) else str(text)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text})

    details = []
    ok = True
    for check in case["checks"]:
        passed, label = check(text)
        details.append(f"{'✓' if passed else '✗'} {label}")
        ok = ok and passed
    if verbose:
        print(f"\n--- {case['id']} ---\n{text}\n")
    return ok, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="问答级评估集")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--verbose", action="store_true", help="打印完整回答")
    parser.add_argument("--only", help="只跑指定 id，逗号分隔")
    args = parser.parse_args(argv)

    print("计算标准答案…")
    truth = ground_truth()
    print(f"  最新 ONI {truth['latest']}（{truth['label']}），环比 {truth['change']}")
    print(f"  最强暖事件 {truth['warm_top']['peak_label']} {truth['warm_top']['peak']}")
    print(f"  最强冷事件 {truth['cold_top']['peak_label']} {truth['cold_top']['peak']}")
    print(f"  分型 {truth['type']}（{truth['type_period']}）\n")

    cases = build_cases(truth)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    agent = build_agent()
    passed = 0
    failures = []
    for index, case in enumerate(cases, start=1):
        try:
            ok, details = run_case(agent, case, args.verbose)
        except Exception as exc:
            ok, details = False, [f"✗ 执行异常 {type(exc).__name__}: {exc}"]
        passed += int(ok)
        print(f"[{index:02d}/{len(cases)}] {'PASS' if ok else 'FAIL'} {case['id']}: {case['question'][:34]}")
        if not ok:
            for line in details:
                if line.startswith("✗"):
                    print(f"       {line}")
            failures.append((case["id"], details))

    print(f"\n合计 {len(cases)} 条，通过 {passed}，失败 {len(cases) - passed}"
          f"，通过率 {passed / max(1, len(cases)):.0%}")
    if failures:
        print("失败明细：")
        for case_id, details in failures:
            print(f"  - {case_id}: " + "；".join(details))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
