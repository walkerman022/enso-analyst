#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""气象工具的错误文案。

为什么不用框架自带的 LA-503：它的文案是 "The model provider is temporarily
unavailable"，把 NOAA 挂掉说成"模型服务不可用"，模型会去重试或换模型，
而不是告诉用户"数据源查不到"。这里保留 [LA- 前缀（框架靠它识别错误），
正文改成数据源语义。
"""

from __future__ import annotations


def data_source_error(exc: BaseException, source: str) -> str:
    text = str(exc)
    if "timeout" in text.lower() or "timed out" in text.lower():
        headline = f"读取 {source} 超时"
    elif "HTTP" in text:
        headline = f"{source} 返回异常状态"
    else:
        headline = f"{source} 当前不可访问"
    return (
        f"[LA-TOOL] {headline}。"
        f"Action: 告知用户数据源暂时不可用，说明未能取到数据；不要凭记忆给数值。"
        f"Details: {type(exc).__name__}: {text}"
    )


def tool_error(exc: BaseException, action: str) -> str:
    return (
        f"[LA-TOOL] {action} 失败。"
        f"Action: 说明该步骤未完成，不要编造结果。"
        f"Details: {type(exc).__name__}: {exc}"
    )
