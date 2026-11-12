#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 组装：所有"配置"集中在这里，别的文件只放能力。

三处关键配置（对应清单里的 W2 / W2b / W3）：
- ContextBudget：不传就永不压缩（core.py:1200 有 if not self.context_budget: return）；
  chars_per_token 默认 4.0 是按英文估的，中文实际约 1.5-2 字符/token，不调会低估预算。
- ContextCompactor：max_inline_tool_chars 默认 12000，ONI 全表能到 2 万字符，收到 4000。
- instructions：框架没有独立 system_prompt 入口，身份和约束只能走这个参数（core.py:725）。
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from LightAgent import LightAgent
from LightAgent import BudgetLimits
from LightAgent.session import ContextBudget, ContextCompactor

from weather_agent import evidence, memory, prompts
from weather_agent.credentials import load_credentials
from weather_agent.number_check import make_input_guardrail, make_output_guardrail
from weather_agent.summarizer import make_summarizer
from weather_agent.tools import ALL_TOOLS


logger = logging.getLogger(__name__)

# 这些工具本来就不产生"需要回查的数值"：出图返回路径、偏好返回动作、模型清单返回枚举。
# 给它们补记只会污染账本，所以明确豁免。
EVIDENCE_EXEMPT = {
    "plot_enso_trend", "plot_enso_regions",
    "save_preference", "recall_preferences", "forget_preference",
    "list_prediction_models", "list_enso_history",
}


def with_evidence_safety_net(func):
    """兜底记账：工具返回时若账本没有新增，就把结果里的数值补记进去。

    为什么不直接用装饰器替代手写 record：
    有些工具需要逐行打时间标签（如 fetch_oni），通用装饰器做不到；
    所以保留手写记账为主，这层只兜"新工具忘了调 record"的情况。
    补记的 source 是 tool/<name>、没有对象标签，只能做"值存在"校验，
    但比完全查不到出处强。
    """
    info = getattr(func, "tool_info", {}) or {}
    name = info.get("tool_name") or getattr(func, "__name__", "tool")
    if name in EVIDENCE_EXEMPT:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        before = evidence.count()
        result = func(*args, **kwargs)
        if isinstance(result, dict) and evidence.count() == before:
            added = evidence.record(f"tool/{name}", result)
            if added:
                logger.warning("tool %s returned numbers without recording evidence; "
                               "auto-recorded %d entries", name, added)
        return result

    wrapper.tool_info = info
    return wrapper


def estimate_tool_tokens(tools: list, chars_per_token: float = 1.8) -> int:
    """估算工具 schema 占多少 token。

    为什么不能等注册完再算：预算要在构造时传给 Agent。
    这里按 tool_info 近似（名称 + 描述 + 参数 + 包装），再留 10% 余量。
    """
    total_chars = 0
    for func in tools:
        info = getattr(func, "tool_info", {}) or {}
        total_chars += len(json.dumps(info, ensure_ascii=False)) + 60   # 60 ≈ type/function 包装
    return int(total_chars / chars_per_token * 1.1) + 1


def build_agent(
        *,
        guardrail_mode: str = "annotate",
        guardrail_scope: str = "default",
        debug: bool = False,
        tools: list[Any] | None = None,
        session_store: Any = None,
        name: str | None = None,
        instructions: str | None = None,
        role: str | None = None,
        model_calls: int = 20,
        tool_calls: int = 30,
        request_timeout: float = 90.0,
        max_input_chars: int = 4000,
        enable_summary: bool = True,
) -> LightAgent:
    """组装一个可用的 ENSO 助手。

    guardrail_mode:
        annotate —— 发现无出处数字时在回答末尾加提示（默认）
        block    —— 直接拦下（验收测试用）
        off      —— 关闭（排查模型行为时用）
    session_store:
        传 SqliteSessionStore 后，配合 run(session_id=...) 可跨进程保留对话。
    """
    credentials = load_credentials()
    # 长期记忆：把已保存的偏好拼进系统提示（只有几条，不需要检索）
    base_instructions = instructions or prompts.INSTRUCTIONS
    preference_block = memory.preference_block()
    if preference_block:
        base_instructions = f"{base_instructions}\n\n{preference_block}"
    final_tools = [with_evidence_safety_net(item) for item in (tools if tools is not None else ALL_TOOLS)]
    return LightAgent(
        name=name or prompts.NAME,
        instructions=base_instructions,
        role=role or prompts.ROLE,
        model=credentials["model"],
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        tools=final_tools,
        session_store=session_store,
        # 气象助手不用 SKILL.md 那套技能机制，关掉自动发现可以少发 4 个 schema，省 token
        auto_discover_skills=False,
        # 32000 是"够用且留余量"的值：DeepSeek 上下文远大于此，
        # 我们主动设小是为了让压缩在长会话里真的会触发，而不是等到爆掉。
        context_budget=ContextBudget(
            max_tokens=32000,
            reserved_output_tokens=4096,
            chars_per_token=1.8,        # 用来算token的，中文实测约 1.5-2 字符/token
            # 工具 schema 每轮都要发，但不是 messages，必须单独预留；
            # 不预留的话预算会系统性偏乐观（19 个工具实测约 3.5k token）。
            reserved_tools_tokens=estimate_tool_tokens(final_tools),
        ),
        # 12000 是实测出来的：工具单次最多返 100 季节行 × 4 字段（包含：seas（季节标签）、yr（年份）、total（绝对海温）、anom（ONI 距平））
        # JSON 约 6.4k 字符，留近一倍余量。设太小（如 4000）会把正常取数溢出成哈希，模型就读不到数据。
        # 摘要器：只在“必须丢旧消息”时被调用，正常问答不会触发额外开销。
        # 摘要里禁止出现阿拉伯数字，避免改写数值后与证据账本对不上。
        context_compactor=ContextCompactor(
            summarizer=make_summarizer(timeout=request_timeout) if enable_summary else None,
            max_inline_tool_chars=12000,
        ),
        # 硬上限：一次问答通常 2-6 次模型调用、3-8 次工具调用。
        # 设 20/30 是为了拦住"模型反复查同一份数据"的死循环，而不是限制正常使用。
        budget_limits=BudgetLimits(model_calls=model_calls, tool_calls=tool_calls, seconds=180),
        # SDK 默认超时是 600 秒，遇到卡住的连接会挂 10 分钟；这里收到 90 秒
        client_timeout=request_timeout,
        input_guardrails=[make_input_guardrail(max_input_chars)],
        output_guardrails=[make_output_guardrail(guardrail_mode, guardrail_scope)],
        debug=debug,
    )
