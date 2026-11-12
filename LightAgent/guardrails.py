#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightweight guardrail primitives for input, tool-call, and output checks.
护栏是钩子的窄化版：只保留"检查"这个职责，砍掉所有流程控制。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GuardrailDecision:
    """Decision returned by an input, tool, or output guardrail."""
    """告诉框架"放不放行"，以及"要不要换成这个值"""

    allowed: bool               # True 放行，False 拦截。必填
    reason: str | None = None   # 拦截理由，会出现在 [LA-GUARDRAIL] 错误里
    value: Any = None           # 替换后的值。None 表示"不改"


class GuardrailManager:
    """Runs configured guardrails while keeping the default runtime unchanged."""
    """构造时传三组护栏函数,每次检查返回一个 GuardrailDecision, 按位置分组存放护栏，检查时只跑对应那一组 """

    def __init__(
            self,
            *,
            input_guardrails: list[Callable[..., Any]] | None = None,
            tool_guardrails: list[Callable[..., Any]] | None = None,
            output_guardrails: list[Callable[..., Any]] | None = None,
    ):  # 三个列表完全独立，互相都看不到对方的内容。
        self.input_guardrails = input_guardrails or []          # 检查用户输入的
        self.tool_guardrails = tool_guardrails or []            # 检查工具调用的
        self.output_guardrails = output_guardrails or []        # 检查模型输出的

    def check_input(self, query: str, context: dict[str, Any] | None = None) -> GuardrailDecision:
        return self._run_guardrails(self.input_guardrails, query, context or {})

    def check_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        payload = {"tool_name": tool_name, "arguments": arguments}   # 内部拼成字典
        return self._run_guardrails(self.tool_guardrails, payload, context or {})

    def check_output(self, output: str, context: dict[str, Any] | None = None) -> GuardrailDecision:
        return self._run_guardrails(self.output_guardrails, output, context or {})
    """三个方法最终都调用_run_guardrails, 只是传入的列表和值不同"""

    '''依次跑护栏，任何一个拦截就停，否则就把改写后的值传给下一个'''
    def _run_guardrails(
            self,
            guardrails: list[Callable[..., Any]],
            value: Any,
            context: dict[str, Any],
    ) -> GuardrailDecision:
        current_value = value
        for guardrail in guardrails:
            try:
                raw_decision = guardrail(current_value, context)
            except Exception as exc:
                # fail-closed：护栏自己崩了必须拦截，不能因为故障而放行
                return GuardrailDecision(
                    False,
                    reason=f"guardrail crashed: {type(exc).__name__}: {exc}",
                )
            decision = self._coerce_decision(raw_decision, current_value)   # 归一化
            if not decision.allowed:
                return decision                                             # 拦截：立刻返回
            if decision.value is not None:
                current_value = decision.value                         # 链式改写：更新当前值，下一个护栏收到的上一个改动过的值
        return GuardrailDecision(allowed=True, value=current_value)         # 全部通过，返回current_value

    '''不要求你构造 decision，返回 True / False / 字符串都行'''
    @staticmethod
    def _coerce_decision(raw_decision: Any, current_value: Any) -> GuardrailDecision:
        if isinstance(raw_decision, GuardrailDecision):
            return raw_decision
        if raw_decision is None or raw_decision is True:
            return GuardrailDecision(allowed=True, value=current_value)
        if raw_decision is False:
            return GuardrailDecision(allowed=False, reason="Guardrail blocked this operation.")
        if isinstance(raw_decision, str):
            return GuardrailDecision(allowed=False, reason=raw_decision)
        if isinstance(raw_decision, dict):
            return GuardrailDecision(
                allowed=bool(raw_decision.get("allowed", True)),
                reason=raw_decision.get("reason"),
                value=raw_decision.get("value", current_value),
            )

        # 返回一个无法识别的类型（比如数字、列表），会被当成"这是新值"，放行并替换。既是灵活性也是风险
        return GuardrailDecision(allowed=True, value=raw_decision)


DEFAULT_PRIVACY_PATTERNS = (
    r"\b\d{3}-\d{2}-\d{4}\b",                                     # 美国社保号
    r"\b(?:\d[ -]*?){13,19}\b",                                   # 信用卡/身份证
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",                 # 邮箱
    r"\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+",      # 密钥/密码赋值
)


# 检测四类隐私数据，命中就拦截
# 用法：LightAgent(..., input_guardrails=[privacy_input_guardrail()])
def privacy_input_guardrail(
        patterns: list[str] | tuple[str, ...] | None = None,
        *,
        reason: str = "privacy-sensitive input detected",
) -> Callable[[str, dict[str, Any]], GuardrailDecision]:
    """Return an input guardrail that blocks common private data patterns."""
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in (patterns or DEFAULT_PRIVACY_PATTERNS)]

    def guardrail(query: str, context: dict[str, Any]) -> GuardrailDecision:
        text = "" if query is None else str(query)
        if any(pattern.search(text) for pattern in compiled):
            return GuardrailDecision(False, reason=reason)
        return GuardrailDecision(True, value=query)

    return guardrail


# 列出的工具在未批准前，一律不许调用
# 缺点：approved在构造护栏时就已经确定了，运行期间无法修改。想“这一次批准、下一次不批准”，做不到——必须重建 Agent 或换护栏对象。
def sensitive_tool_confirmation_guardrail(
        tool_names: list[str] | tuple[str, ...],            # 要保护的工具名列表
        *,
        approved: bool = False,
        reason: str = "tool call requires approval",        # 是否已批准（静态开关）
) -> Callable[[dict[str, Any], dict[str, Any]], GuardrailDecision]:
    """Return a tool guardrail that blocks selected tools until approved."""
    sensitive = {str(name) for name in tool_names}

    def guardrail(tool_call: dict[str, Any], context: dict[str, Any]) -> GuardrailDecision:
        if tool_call.get("tool_name") in sensitive and not approved:
            return GuardrailDecision(False, reason=reason)          # 命中且在保护名单，approved=False -> 拦截
        return GuardrailDecision(True, value=tool_call)             # 其它 -> 执行

    return guardrail


# 检查某个参数的值是否符合规则。
def high_risk_parameter_guardrail(
        rules: dict[str, Callable[[Any], bool] | list[str] | tuple[str, ...]],
        *,
        reason: str = "high-risk tool parameter blocked",
) -> Callable[[dict[str, Any], dict[str, Any]], GuardrailDecision]:
    """Return a tool guardrail for validating sensitive argument values."""

    def guardrail(tool_call: dict[str, Any], context: dict[str, Any]) -> GuardrailDecision:
        arguments = tool_call.get("arguments") or {}                            # ① 取参数，空则空字典
        for name, rule in rules.items():                                        # ② 逐条规则检查
            value = arguments.get(name)                                         # ③ 取这个参数的值
            # 参数没传时，arguments.get(name) 返回 None，rule(None)直接抛异常，护栏没有 try/except，异常会冒到 run()，从而会直接对话崩溃
            # 模型漏传一个参数 → 护栏抛 TypeError → 整个对话崩，而不是返回 [LA-GUARDRAIL]。
            if callable(rule) and not rule(value):                              # ④ 函数规则：返回 False 就拦
                return GuardrailDecision(False, reason=f"{reason}: {name}")
            if isinstance(rule, (list, tuple)) and value not in rule:           # ⑤ 白名单：不在里面就拦
                return GuardrailDecision(False, reason=f"{reason}: {name}")
        return GuardrailDecision(True, value=tool_call)

    return guardrail


# 把输出里的隐私数据替换成 [redacted]，永远放行。
def output_redaction_guardrail(
        patterns: list[str] | tuple[str, ...] | None = None,
        *,
        replacement: str = "[redacted]",
) -> Callable[[str, dict[str, Any]], GuardrailDecision]:
    """Return an output guardrail that redacts common sensitive patterns."""
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in (patterns or DEFAULT_PRIVACY_PATTERNS)]

    def guardrail(output: str, context: dict[str, Any]) -> GuardrailDecision:
        text = "" if output is None else str(output)
        for pattern in compiled:
            text = pattern.sub(replacement, text)
        return GuardrailDecision(True, value=text)

    return guardrail
