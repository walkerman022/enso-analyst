#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""证据账本：记录本轮对话中工具真实返回过的每一个数值。

为什么需要它：
LightAgent 的输出护栏只收到 {"agent_name", "trace_id"}（core.py:1432），
拿不到 chat_params 里的工具消息，所以"模型说的数字有没有出处"这件事
不能靠回读 messages，只能靠工具自己在返回前登记。

为什么要分 scope：
会商模式里同时有海洋组、大气组、历史组三个 Agent。若共用一个账本，
A 组查到的数值会被 B 组"借用"，回查就失去意义。每个角色一个命名空间，
互不可见；首席只看三份报告，不用账本。

用法：
    evidence.reset("ocean")                       # 开会前清空该角色账本
    evidence.record("ONI/NOAA", payload, scope="ocean")
    evidence.match(2.16, 0.005, scope="ocean")
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

DEFAULT_SCOPE = "default"

# 当前生效的账本命名空间。工具里不需要显式传 scope，
# 由编排层在调用某个角色前 use_scope("ocean") 切过去。
_ACTIVE_SCOPE: ContextVar[str] = ContextVar("active_scope", default=DEFAULT_SCOPE)

# scope -> 条目列表。条目结构：数值 + 数据源 + 字段路径 + 所属对象（时间标签）
_LEDGERS: dict[str, list[dict[str, Any]]] = {}

# 单个 scope 的条目上限：超出就丢最旧的（FIFO）。
# 不设上限的后果是长会话内存只增不减、回查全量扫描越来越慢；
# 丢最旧的代价是"很早之前查过的数不再被认作有出处"——那本来就该重查。
MAX_PER_SCOPE = 2000

# 知识库引用：每个 scope 记下"本轮召回过哪些片段 id"
_CITATIONS: dict[str, set[str]] = {}

# 只识别"整串都是数字"的字符串，避免把 "2026-10-03" 之类的日期当数字吞进来
_NUMERIC_STRING = re.compile(r"^-?\d+(?:\.\d+)?$")


def _ledger(scope: str) -> list[dict[str, Any]]:
    return _LEDGERS.setdefault(scope, [])


def record_citations(ids: list[str], scope: str | None = None) -> int:
    """登记本轮召回的知识片段 id，供回答里的 [kb:xxx] 校验。"""
    scope = scope or _ACTIVE_SCOPE.get()
    bucket = _CITATIONS.setdefault(scope, set())
    before = len(bucket)
    bucket.update(str(item) for item in ids)
    return len(bucket) - before


def cited_ids(scope: str | None = None) -> set[str]:
    return set(_CITATIONS.get(scope or _ACTIVE_SCOPE.get(), set()))


def use_scope(scope: str):
    """切换当前账本命名空间，返回 token（可传给 reset_scope 还原）。"""
    return _ACTIVE_SCOPE.set(scope)


def reset_scope(token) -> None:
    _ACTIVE_SCOPE.reset(token)


def current_scope() -> str:
    return _ACTIVE_SCOPE.get()


def merge_into(target: str, sources: list[str]) -> int:
    """把多个角色的账本合并进一个（首席复核用）。返回合并后的条数。"""
    merged = _ledger(target)
    for source in sources:
        merged.extend(_ledger(source))
    return len(merged)


def reset(scope: str | None = DEFAULT_SCOPE) -> None:
    """清空账本。scope=None 表示清空所有命名空间（测试和开会前用）。"""
    if scope is None:
        _LEDGERS.clear()
        _CITATIONS.clear()
        return
    _ledger(scope).clear()
    _CITATIONS.pop(scope, None)


def scopes() -> list[str]:
    return sorted(_LEDGERS)


def record(
        source: str,
        payload: Any,
        *,
        field: str = "",
        object_label: str | None = None,
        scope: str | None = None,
) -> int:
    """递归登记 payload 里所有数值，返回登记条数。

    source 用来在报错/调试时说明出处，例如 "ONI/NOAA CPC"。
    field 是可选的字段前缀，递归时会自动拼接（如 data[0].anom）。
    object_label 标记"这个数值属于哪个对象"（如 "JAS 2026"、"2026-06"）。
    scope 是命名空间，多 Agent 时每个角色一个。
    """
    scope = scope or _ACTIVE_SCOPE.get()
    ledger = _ledger(scope)
    count = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            count += record(
                source, value,
                field=f"{field}.{key}" if field else str(key),
                object_label=object_label,
                scope=scope,
            )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            count += record(source, value, field=f"{field}[{index}]", object_label=object_label, scope=scope)
    elif isinstance(payload, bool):
        # bool 是 int 的子类，必须显式排除，否则 True 会被记成 1
        pass
    elif isinstance(payload, (int, float)):
        ledger.append({"value": float(payload), "source": source, "field": field, "object": object_label})
        if len(ledger) > MAX_PER_SCOPE:
            del ledger[: len(ledger) - MAX_PER_SCOPE]
        count += 1
    elif isinstance(payload, str) and _NUMERIC_STRING.match(payload.strip()):
        ledger.append({"value": float(payload.strip()), "source": source, "field": field, "object": object_label})
        if len(ledger) > MAX_PER_SCOPE:
            del ledger[: len(ledger) - MAX_PER_SCOPE]
        count += 1
    return count


def count(scope: str | None = None) -> int:
    """当前账本条目数。给"漏记兜底"用：前后对比就知道工具有没有记账。"""
    return len(_ledger(scope or _ACTIVE_SCOPE.get()))


def entries(scope: str | None = None) -> list[dict[str, Any]]:
    """返回账本副本，供调试和测试使用。"""
    return list(_ledger(scope or _ACTIVE_SCOPE.get()))


def values(scope: str | None = None) -> list[float]:
    return [item["value"] for item in _ledger(scope or _ACTIVE_SCOPE.get())]


def match(number: float, tolerance: float, scope: str | None = None) -> dict[str, Any] | None:
    """在账本里找与 number 相差不超过 tolerance 的记录，找不到返回 None。"""
    best: dict[str, Any] | None = None
    best_diff = tolerance
    for item in _ledger(scope):
        diff = abs(item["value"] - number)
        if diff <= best_diff:
            best = item
            best_diff = diff
    return best


def match_for_object(
        number: float,
        object_label: str,
        tolerance: float,
        scope: str | None = None,
) -> dict[str, Any] | None:
    """只在指定对象的记录里找。用于"2.16 是不是 JAS 2026 的值"。"""
    scope = scope or _ACTIVE_SCOPE.get()
    best: dict[str, Any] | None = None
    best_diff = tolerance
    for item in _ledger(scope):
        if item.get("object") != object_label:
            continue
        diff = abs(item["value"] - number)
        if diff <= best_diff:
            best = item
            best_diff = diff
    return best


def other_objects_for(
        number: float,
        tolerance: float,
        limit: int = 3,
        scope: str | None = None,
) -> list[str]:
    """返回"哪些对象下确实有这个数值"。

    用途：值是真的，但可能挂错了年份。返回的标签用来告诉模型"2.16 属于 JAS 2026"。
    """
    scope = scope or _ACTIVE_SCOPE.get()
    labels: list[str] = []
    for item in _ledger(scope):
        label = item.get("object")
        if not label or label in labels:
            continue
        if abs(item["value"] - number) <= tolerance:
            labels.append(label)
            if len(labels) >= limit:
                break
    return labels


def mismatched_objects(
        number: float,
        stated: str,
        tolerance: float,
        scope: str | None = None,
        limit: int = 3,
) -> list[str]:
    """返回"同一种量、但属于别的对象"的标签。

    只在两边字段相同时才算错配：否则"SOI 0.30（2025-06）"会被误判成
    "ASO 2026 的 ONI 写错了"。字段就是 record 的 field（anom / total / soi / mean_anom…）。
    """
    scope = scope or _ACTIVE_SCOPE.get()
    ledger = _ledger(scope)
    stated_fields = {item["field"] for item in ledger if item.get("object") == stated}
    if not stated_fields:
        return []
    labels: list[str] = []
    for item in ledger:
        label = item.get("object")
        if not label or label == stated or label in labels:
            continue
        if item["field"] not in stated_fields:
            continue
        if abs(item["value"] - number) <= tolerance:
            labels.append(label)
            if len(labels) >= limit:
                break
    return labels


def match_derived(
        number: float,
        tolerance: float,
        max_entries: int = 300,
        scope: str | None = None,
) -> dict[str, Any] | None:
    """允许"由账本数值算出来的量"。

    实测踩到的例子：模型写"比上一季高 0.36°C，即 2.16-0.36=1.80"，
    1.80 从来没被任何工具直接返回过，但它确实是账本里两个数的差。
    这种派生量不该判成幻觉，否则每轮追问都会被误标。

    只做加减，不做乘除：乘除容易凑出巧合值，会把真正编造的数字放过去。
    """
    scope = scope or _ACTIVE_SCOPE.get()
    values = [item["value"] for item in _ledger(scope)[:max_entries]]
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            for expression, result in (
                    (f"{left:g}-{right:g}", left - right),
                    (f"{right:g}-{left:g}", right - left),
                    (f"{left:g}+{right:g}", left + right),
            ):
                if abs(result - number) <= tolerance:
                    return {"value": number, "source": "derived", "field": expression}
    return None


def describe(limit: int = 8, scope: str | None = None) -> str:
    """账本摘要，写进错误信息里方便排查。"""
    scope = scope or _ACTIVE_SCOPE.get()
    ledger = _ledger(scope)
    if not ledger:
        return f"（账本[{scope}]为空：还没有调用过取数工具）"
    parts = [f"{item['value']:g}({item['source']})" for item in ledger[:limit]]
    more = "" if len(ledger) <= limit else f" …共 {len(ledger)} 个"
    return "、".join(parts) + more
