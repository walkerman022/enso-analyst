#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数值回查：判断模型回答里的数字有没有工具出处。

这是防幻觉的第三层，也是唯一能"判定"的一层：
前两层（返回带 source、prompt 要求引用）只是约束，模型可以不听；这一层是事后核对。

LightAgent 的输出护栏只拿到 {"agent_name", "trace_id"}（core.py:1424），
拿不到 messages，所以比对对象是 evidence 账本，而不是对话历史。

三类数字的处理：
1. 账本里能查到 → 通过；
2. 领域常量（Niño3.4、±0.5、5°N-5°S 这类）→ 通过，否则会大面积误杀；
3. 其余小数 → 判为"无出处"，默认在回答末尾加提示，严格模式直接拦截。

整数单独处理：年份、月份、季节数这类整数噪声太多，只记为"弱证据"，不拦。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from LightAgent.guardrails import GuardrailDecision

from weather_agent import evidence

# URL 里的数字（端口、路径年份）不参与回查
_URL_PATTERN = re.compile(r"https?://\S+")

# 领域标识里的数字：Niño3.4 / Niño 3 / 5°N / 120°W 这些是术语不是数据
_IDENTIFIER_PATTERNS = (
    re.compile(r"Ni[nñ]o\s*3\.4", re.IGNORECASE),
    re.compile(r"Ni[nñ]o\s*[34]\b", re.IGNORECASE),
    re.compile(r"\bNINO\s*3\.4\b", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*°?\s*[NSEW]\b"),
)

# 预置的领域常量：阈值、区域范围、季节数、常见百分比
DOMAIN_CONSTANTS = {
    0.5, 5.0, 3.4, 4.0, 12.0, 120.0, 170.0,
    1.5, 2.0, 2.5, 30.0, 90.0, 100.0, 3.0,
}

# 数字本身（前面不能紧贴字母或小数点，避免把 abc12 / 1.2.3 拆开）
_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-−+]?\d+(?:\.\d+)?")

_YEAR_MIN, _YEAR_MAX = 1900, 2100

# 知识库引用：[kb:def-oni]、[kb:cpc-faq-003]
_CITATION = re.compile(r"\[kb:([A-Za-z0-9._-]+)\]")
_WEB_CITATION = re.compile(r"\[web:([A-Za-z0-9._-]+)\]")

# ---- 值-对象绑定：判断"这个数有没有配错年份/季节" ----

# 句子切分：中文标点和换行都算断句
_CLAUSE_SPLIT = re.compile(r"[。；;！!？?\n]+")

# 显式季节标签：JAS 2026、NDJ 1997
_SEASON_LABEL = re.compile(
    r"\b(DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# 不带年份的季节词，用来判断"这句话是否在比较两个时期"
_SEASON_TOKEN = re.compile(r"\b(DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\b", re.IGNORECASE)
_COMPARISON_WORDS = ("上一季", "上季", "前季", "前一期", "上一期", "上一月", "上月", "环比")

# 月份区间表述：2026 年 7-9 月 → 中心月 8 → JAS
_RANGE_LABEL = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*[-–—~至到]\s*(\d{1,2})\s*月")

# 中心月 → 季节标签（与 enso_fetch.SEASONS 同序，这里独立定义避免循环导入）
_SEASON_BY_CENTER_MONTH = {
    1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
    7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
}


@dataclass
class CheckResult:
    ok: bool
    verified: list[dict[str, Any]] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    weak: list[str] = field(default_factory=list)
    unverified: list[dict[str, Any]] = field(default_factory=list)
    # 值真实存在，但挂到了别的时间/对象上（"2.16 是 JAS 2026 的，被说成 2015 年"）
    mismatched: list[dict[str, Any]] = field(default_factory=list)
    # 引用校验：回答里写了 [kb:xx]，就要求 xx 在本次召回集里
    citations: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"verified={len(self.verified)} constants={len(self.constants)} "
            f"weak={len(self.weak)} unverified={len(self.unverified)} "
            f"mismatched={len(self.mismatched)} citations={len(self.citations)} "
            f"bad_citations={len(self.bad_citations)}"
        )


def _tolerance(decimals: int) -> float:
    """允许四舍五入：两位小数容差 0.005，一位 0.05。

    整数不给宽容量（0.01），否则"2°C"会把 2.16 蒙过去。
    末尾加一个极小量：2.45 存成浮点后与 2.4 的差是 0.05000000000000027，
    不加这个量会把正常四舍五入判成编造。
    """
    if decimals <= 0:
        return 0.01
    return 0.5 * (10 ** -decimals) + 1e-9


def _prepared(text: str) -> str:
    """去掉 URL 和领域标识，避免它们里面的数字被当成数据。"""
    cleaned = _URL_PATTERN.sub(" ", text)
    for pattern in _IDENTIFIER_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def extract_numbers(text: str) -> list[str]:
    return [match.group(0) for match in _NUMBER_PATTERN.finditer(_prepared(text))]


def _clauses(text: str) -> list[str]:
    """按句号/分号/问号/换行断句。绑定校验只在句子内部做，跨句容易误配。"""
    return [part.strip() for part in _CLAUSE_SPLIT.split(text) if part.strip()]


def labels_in(clause: str) -> list[str]:
    """找出这句话里提到的时间标签，统一成 "JAS 2026" 这种形式。

    支持两种写法：显式季节（JAS 2026）和月份区间（2026 年 7-9 月 → 中心月 8 → JAS）。
    """
    labels: list[str] = []
    for match in _SEASON_LABEL.finditer(clause):
        labels.append(f"{match.group(1).upper()} {match.group(2)}")
    for match in _RANGE_LABEL.finditer(clause):
        year, start = match.group(1), int(match.group(2))
        season = _SEASON_BY_CENTER_MONTH.get(start + 1)
        if season:
            labels.append(f"{season} {year}")
    return list(dict.fromkeys(labels))


def verify(text: str, scope: str = "default") -> CheckResult:
    """核对一段文本里的数字（含值-对象绑定）。scope 决定查哪个账本。"""
    result = CheckResult(ok=True)
    recalled = evidence.cited_ids(scope)
    for cited in _CITATION.findall(text or ""):
        if cited in result.citations:
            continue
        result.citations.append(cited)
        if cited not in recalled:
            result.bad_citations.append(cited)
            result.ok = False
    for cited in _WEB_CITATION.findall(text or ""):
        key = f"web:{cited}"
        if key in result.citations:
            continue
        result.citations.append(key)
        if key not in recalled:
            result.bad_citations.append(key)
            result.ok = False

    for clause in _clauses(text):
        # 只有"这句只提到一个时间标签"时才做绑定校验；
        # 一句里出现多个标签（比较句）无法确定数字归属，跳过而不是猜。
        clause_labels = labels_in(clause)
        # 句内出现两个及以上季节词（含不带年份的），或是比较句式，就不做绑定：
        # 这种句子里数值本来就来自不同时期（"较上一季 JAS 的 2.16 再升 0.30"）。
        season_tokens = {match.group(1).upper() for match in _SEASON_TOKEN.finditer(clause)}
        is_comparison = any(word in clause for word in _COMPARISON_WORDS)
        stated = clause_labels[0] if (len(clause_labels) == 1 and len(season_tokens) <= 1 and not is_comparison) else None

        numbers = extract_numbers(clause)
        # 双标签句：按“先出现的数字配先出现的标签”做配对校验。
        # 不做全排列搜索——那会把调换后的错误配对也判成通过。
        decimal_numbers = [item for item in numbers if "." in item]
        if len(clause_labels) == 2 and len(decimal_numbers) == 2:
            pairs = list(zip(decimal_numbers, clause_labels))
            matched = []
            for raw, label in pairs:
                value = float(raw.replace("−", "-").lstrip("+"))
                hit = evidence.match_for_object(value, label, _tolerance(2), scope=scope)
                matched.append(bool(hit))
            if all(matched):
                for raw, label in pairs:
                    value = float(raw.replace("−", "-").lstrip("+"))
                    hit = evidence.match_for_object(value, label, _tolerance(2), scope=scope)
                    result.verified.append({"value": value, "source": hit["source"],
                                            "field": hit["field"], "object": label})
                continue
            cross = list(zip(decimal_numbers, list(reversed(clause_labels))))
            if all(evidence.match_for_object(float(raw.replace("−", "-").lstrip("+")), label,
                                             _tolerance(2), scope=scope)
                   for raw, label in cross):
                for raw, label in pairs:
                    value = float(raw.replace("−", "-").lstrip("+"))
                    others = evidence.mismatched_objects(value, label, _tolerance(2), scope=scope)
                    if others:
                        result.mismatched.append({"raw": raw, "stated": label, "actual": others[0]})
                result.ok = False
                continue

        for raw in numbers:
            normalized = raw.replace("−", "-").lstrip("+")
            try:
                value = float(normalized)
            except ValueError:
                continue
            decimals = len(normalized.split(".", 1)[1]) if "." in normalized else 0
            tolerance = _tolerance(decimals)

            # 常量优先判断：0.5 是判据阈值，不能因为账本里有个 0.46 就被"近似匹配"成数据
            if value in DOMAIN_CONSTANTS or abs(value) in DOMAIN_CONSTANTS:
                result.constants.append(normalized)
                continue

            # 值-对象绑定：只对小数做。
            # 整数多为序号、月份、季数（"1."、"8-10 月"、"5 季"），绑上去会大面积误报，
            # 而且整数本来就只记弱证据、不拦截。
            if stated is not None and decimals > 0:
                hit_object = evidence.match_for_object(value, stated, tolerance, scope=scope)
                if hit_object is not None:
                    result.verified.append({
                        "value": value,
                        "source": hit_object["source"],
                        "field": hit_object["field"],
                        "object": stated,
                    })
                    continue
                others = evidence.mismatched_objects(value, stated, tolerance, scope=scope)
                if others:
                    result.mismatched.append({
                        "raw": normalized,
                        "stated": stated,
                        "actual": others[0],
                    })
                    result.ok = False
                    continue

            hit = evidence.match(value, tolerance, scope=scope)
            if hit is not None:
                result.verified.append({"value": value, "source": hit["source"], "field": hit["field"]})
                continue

            # 派生量：账本里两个数相减/相加得到的值也算有出处
            derived = evidence.match_derived(value, tolerance, scope=scope)
            if derived is not None:
                result.verified.append(
                    {"value": value, "source": "derived", "field": derived["field"]}
                )
                continue

            if decimals == 0:
                if _YEAR_MIN <= abs(value) <= _YEAR_MAX:
                    # 年份不算数据，但也不值得记成弱证据
                    continue
                result.weak.append(normalized)
                continue

            result.unverified.append({"raw": normalized, "value": value})
            result.ok = False
    return result


def _format_unverified(items: list[dict[str, Any]]) -> str:
    shown = "、".join(item["raw"] for item in items[:6])
    if len(items) > 6:
        shown += f" 等 {len(items)} 个"
    return shown


def make_input_guardrail(max_chars: int = 4000):
    """输入护栏：拦超长、空、含控制字符的问题。

    为什么是这三样：它们是"还没到模型就能判定"的硬边界，误报为零；
    提示注入的语义判断做不到零误报，留给输出护栏兜底，这里不越权。
    """

    def guardrail(query: str, context: dict[str, Any] | None = None) -> GuardrailDecision:
        text = query or ""
        stripped = text.strip()
        if not stripped:
            return GuardrailDecision(False, reason="问题为空。Action: 请输入具体问题。")
        if len(text) > max_chars:
            return GuardrailDecision(
                False,
                reason=f"问题超过 {max_chars} 字符（当前 {len(text)}）。Action: 请缩短问题，或分多条提问。",
            )
        # 除换行/制表/回车外的控制字符，常见于粘贴事故与探测 payload
        if any(ord(char) < 32 and char not in "\n\t\r" for char in text):
            return GuardrailDecision(False, reason="问题含控制字符。Action: 请清理后重试。")
        return GuardrailDecision(True)

    return guardrail


def make_output_guardrail(mode: str = "annotate", scope: str = "default"):
    """生成一个可挂到 LightAgent output_guardrails 的函数。

    mode:
        annotate —— 放行，但在回答末尾追加"未经核实"提示（默认，产品可用）
        block    —— 直接拦截，返回 [LA-GUARDRAIL]
        off      —— 只记录不干预（调试用）
    """

    def guardrail(output: str, context: dict[str, Any] | None = None) -> GuardrailDecision:
        result = verify(str(output), scope=scope)
        if result.ok:
            return GuardrailDecision(True)

        parts: list[str] = []
        if result.unverified:
            parts.append(f"无出处：{_format_unverified(result.unverified)}")
        if result.bad_citations:
            shown = "、".join(f"[kb:{item}]" for item in result.bad_citations[:4])
            parts.append(f"引用不存在：{shown}（本轮没有召回这些片段）")
        if result.mismatched:
            # 值是真的但挂错时间，比"凭空编造"更容易骗过人工检查，必须单独提示
            wrong = "、".join(
                f"{item['raw']}（应为 {item['actual']}，却写在 {item['stated']} 名下）"
                for item in result.mismatched[:4]
            )
            parts.append(f"时间错配：{wrong}")
        detail = "；".join(parts)

        if mode == "block":
            return GuardrailDecision(
                False,
                reason=(
                    f"回答含未核实或错配的数值 — {detail}。"
                    f"Action: 只用工具返回值，并确保数值与其季节/年份对应。"
                    f"Ledger: {evidence.describe(4, scope=scope)}"
                ),
            )
        if mode == "off":
            return GuardrailDecision(True)
        patched = (
            f"{output}\n\n> ⚠️ 以下数值需要核对：{detail}。"
            f"如需引用请先查询数据源（{evidence.describe(3, scope=scope)}）。"
        )
        return GuardrailDecision(True, value=patched)

    return guardrail
