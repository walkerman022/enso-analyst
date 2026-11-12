#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具调用记录器：包一层工具，记录实际调用了谁、传了什么参数。

用途：
- 评估工具调用准确率（eval_tools）
- 提示缓存判断是否有副作用（cached_agent）

原来这两处各写了一份，改一处忘一处，所以抽出来。
注意 tool_info 必须显式复制——functools.wraps 复制 __dict__ 通常够用，
但注册器读的是 tool_info 属性，显式赋值最稳。
"""

from __future__ import annotations

import functools
from typing import Any, Callable


class ToolRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def wrap(self, tools: list) -> list:
        wrapped = []
        for tool in tools:
            name = getattr(tool, "tool_info", {}).get("tool_name", tool.__name__)

            @functools.wraps(tool)
            def wrapper(*args, __tool: Callable = tool, __name: str = name, **kwargs):
                self.calls.append((__name, dict(kwargs)))
                return __tool(*args, **kwargs)

            wrapper.tool_info = getattr(tool, "tool_info", {})
            wrapped.append(wrapper)
        return wrapped

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args_of(self, tool_name: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == tool_name]

    def reset(self) -> None:
        self.calls.clear()
