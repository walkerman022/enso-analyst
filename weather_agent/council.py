#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预报会商：三个专家 Agent 各出一份意见，首席 Agent 汇总并指出分歧。

为什么值得多 Agent：
- 单个 Agent 看到 ONI 2.46 之后，后面所有判断都会往"强厄尔尼诺"靠；
- 海洋、大气、历史三套判据需要三份独立上下文，混在一个 prompt 里会互相覆盖；
- 业务上最有价值的一句话是"海洋和大气不一致，置信度中等"——单 Agent 几乎不会主动说。

编排方式：固定流水线（Python 控制），不是自由交接。
    海洋组 → 大气组 → 历史组 → 首席（只读三份报告，不给取数工具）

账本隔离：每个角色一个 scope（ocean / atmosphere / analog / council），
工具里不用改代码，编排层切 evidence.use_scope() 即可。

用法：
    cd D:\\AgentRoom\\LightAgent-0.11.1\\LightAgent-0.11.1
    python -m weather_agent.council "现在厄尔尼诺到什么程度了？"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from weather_agent import evidence
from weather_agent.agent import build_agent
from weather_agent.tools import (
    analog_evolution,
    classify_enso_type,
    describe_enso_state,
    fetch_enso_index,
    fetch_nino_regions,
    fetch_oni,
    list_enso_history,
)
from weather_agent.tools.soi import fetch_soi

# ---- 角色定义 ----

BASE_RULES = """你是 ENSO 会商中的一个专家角色，只负责本领域的事实核查。

通用规则：
1. 所有数值必须来自工具返回，禁止凭记忆或推测；没查就说"未查询"。
2. 每个数值后面用括号注明数据期与来源，例如"ONI +2.46°C（ASO 2026，NOAA CPC）"。
3. 只陈述本领域能看到的事实，不做跨领域推断（例如海洋组不要评价大气环流）。
4. 不预测未来。如果数据只到某个季节，明确说"截至"。
5. 回答控制在 200 字以内，结构固定：结论 / 数值依据 / 数据期与来源 / 本组的不确定性。
"""

OCEAN_RULES = BASE_RULES + """
你的领域：海洋侧 ENSO 指标。
- 用 fetch_enso_index 取最新 ONI，用 describe_enso_state 取官方判据下的状态；
- 用 fetch_nino_regions + classify_enso_type 判断暖中心偏东还是偏中；
- 若分型所用月份与最新 ONI 不同期，必须点明时间差。
"""

ATMOSPHERE_RULES = BASE_RULES + """
你的领域：大气侧响应。
- 用 fetch_soi 取标准化 SOI：持续负值通常对应 El Niño（沃克环流减弱），持续正值对应 La Niña；
- 说明当前 SOI 的符号、量级和连续月数；
- 只能描述"大气是否已经响应"，不要判断海温未来怎么走。
"""

ANALOG_RULES = BASE_RULES + """
你的领域：历史类比。
- 用 list_enso_history 找历史事件，用 analog_evolution 看同类事件峰值之后 3/6/9/12 个季节的演变；
- 必须说明样本量（用了几个历史事件），样本少于 3 个时要明确说"样本有限"；
- 只给历史统计结果，不说"这次一定会怎样"。
"""

CHIEF_RULES = """你是 ENSO 会商的首席，负责汇总三个专家的意见。

规则：
1. 你没有任何取数工具，只能使用三份报告里的数值，禁止自行查询或引入外部数字。
2. 必须显式回答：三组意见一致的地方、不一致的地方、以及你给出结论的置信度。
3. 如果海洋与大气信号不一致，要说明"海气耦合是否完全建立"，并降低置信度。
4. 如果历史类比样本少于 3 个事件，要注明统计不可靠。
5. 输出格式固定，用中文：

## 结论
（一段话，含当前状态、强度、分型）

## 一致点
- ...

## 分歧／缺口
- ...（没有就写"无"）

## 置信度
高 / 中 / 低 —— 一句话理由

## 数据期
（各组结论对应的数据截止时间）
"""


@dataclass
class ExpertSpec:
    key: str
    name: str
    role: str
    instructions: str
    tools: list[Callable[..., Any]] = field(default_factory=list)


EXPERTS: list[ExpertSpec] = [
    ExpertSpec("ocean", "海洋监测组", "海洋侧 ENSO 指标分析", OCEAN_RULES,
               [fetch_enso_index, describe_enso_state, fetch_oni, fetch_nino_regions, classify_enso_type]),
    ExpertSpec("atmosphere", "大气监测组", "大气侧响应分析", ATMOSPHERE_RULES, [fetch_soi]),
    ExpertSpec("analog", "历史类比组", "历史事件类比", ANALOG_RULES,
               [fetch_oni, list_enso_history, analog_evolution]),
]

EXPERT_KEYS = [spec.key for spec in EXPERTS]


def run_expert(spec: ExpertSpec, question: str, *, debug: bool = False) -> dict[str, Any]:
    """跑一个专家角色，返回它的报告和账本条目数。"""
    token = evidence.use_scope(spec.key)
    try:
        agent = build_agent(
            name=spec.name,
            role=spec.role,
            instructions=spec.instructions,
            tools=spec.tools,
            guardrail_mode="annotate",
            guardrail_scope=spec.key,
            model_calls=10,
            tool_calls=15,
            debug=debug,
        )
        text = agent.run(question)
    except Exception as exc:
        # 关键容错：一个专家挂了（限流/超时/预算超限），不能拖垮整场会商。
        # 已跑完的其余专家照常参与汇总，缺席信息在纪要里显式标注。
        text = (
            f"[LA-TOOL] {spec.name}未能产出意见：{type(exc).__name__}。"
            f"Guidance: 该组意见缺失，汇总时必须在『分歧/缺口』里注明本组缺席，不得虚构其结论。"
        )
        return {
            "key": spec.key,
            "name": spec.name,
            "report": text,
            "evidence_count": 0,
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        evidence.reset_scope(token)
    text = text if isinstance(text, str) else str(text)
    return {
        "key": spec.key,
        "name": spec.name,
        "report": text,
        "evidence_count": len(evidence.entries(spec.key)),
        "failed": False,
    }


def _chief_query(question: str, reports: list[dict[str, Any]]) -> str:
    blocks = []
    for item in reports:
        blocks.append(f"### {item['name']}（{item['key']}）\n{item['report']}")
    joined = "\n\n".join(blocks)
    return (
        f"用户的问题：{question}\n\n"
        f"三位专家的报告如下，请按你的输出格式汇总：\n\n{joined}\n\n"
        f"注意：你只能使用以上报告中的数值；报告之间不一致时，必须在“分歧”里指出。"
    )


def run_council(question: str, *, debug: bool = False, verbose: bool = False) -> dict[str, Any]:
    """跑一次会商：三组专家 + 首席。"""
    # 每次会商从干净账本开始，避免上一次的数字被当成出处
    evidence.reset(None)

    reports: list[dict[str, Any]] = []
    for spec in EXPERTS:
        item = run_expert(spec, question, debug=debug)
        reports.append(item)
        if verbose:
            print(f"\n===== {spec.name} =====")
            print(item["report"])

    # 首席的账本是三组的并集：它引用任何一组的数值都算有出处
    evidence.reset("council")
    merged = evidence.merge_into("council", EXPERT_KEYS)

    chief = build_agent(
        name="会商首席",
        role="ENSO 会商汇总",
        instructions=CHIEF_RULES,
        tools=[],              # 关键：不给取数工具，防止它自己另查一遍
        guardrail_mode="annotate",
        guardrail_scope="council",
        model_calls=3,
        tool_calls=0,
        debug=debug,
    )
    verdict = chief.run(_chief_query(question, reports))
    verdict = verdict if isinstance(verdict, str) else str(verdict)

    if verbose:
        print("\n===== 会商首席 =====")
        print(verdict)

    return {
        "question": question,
        "reports": reports,
        "verdict": verdict,
        "evidence": {key: len(evidence.entries(key)) for key in EXPERT_KEYS + ["council"]},
        "chief_evidence": merged,
    }


def format_minutes(result: dict[str, Any]) -> str:
    """把会商结果排成一份可读的纪要（也可以直接 json.dumps）。"""
    lines = [f"# ENSO 会商纪要\n\n**问题**：{result['question']}\n"]
    lines.append("## 首席结论\n")
    lines.append(result["verdict"].strip())
    lines.append("\n## 各组原始意见\n")
    for item in result["reports"]:
        marker = "缺席" if item.get("failed") else f"数值条目 {item['evidence_count']}"
        lines.append(f"### {item['name']}（{marker}）\n")
        lines.append(item["report"].strip())
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ENSO 预报会商（多 Agent）")
    parser.add_argument("query", nargs="*", help="要会商的问题")
    parser.add_argument("--verbose", action="store_true", help="打印每组原始意见")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于程序消费")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    question = " ".join(args.query) or "当前 ENSO 状态如何？海气是否一致？"
    result = run_council(question, debug=args.debug, verbose=args.verbose)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if not args.verbose:
            print(format_minutes(result))
        else:
            print("\n===== 纪要 =====")
            print(format_minutes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
