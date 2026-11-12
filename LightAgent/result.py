#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured result objects for optional LightAgent responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Optional structured return value for non-streaming agent runs."""

    content: str                        # 最终回答文本
    reasoning_content: str | None = None  # 思考过程
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 本次调用过的工具列表
    usage: dict[str, Any] | None = None         # token 用量
    trace_id: str | None = None                 # 追踪号
    trace: list[dict[str, Any]] = field(default_factory=list)       # 完整事件列表
    error: str | None = None                    # 错误信息

    def __str__(self) -> str:
        return self.content
    # 三种result_format对比
    # "str"：只要回答，不要元数据
    # "object"：返回RunResult，要 usage / trace / tool_calls，按属性访问
    # "dict"：要序列化（存文件、发 HTTP）


@dataclass
class StreamEvent:
    """Optional structured event for streaming agent runs."""

    type: str
    data: Any = None
    trace_id: str | None = None

    def __str__(self) -> str:
        return str(self.data)
