#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆工具：让模型能自己读写"用户偏好"。

什么时候该写：用户明确表达长期偏好时才写（"我只看东部型""以后都用中文"），
不要每一轮都写，也不要把当次查询的数值写进去——那是实时数据，存了会过期。
"""

from __future__ import annotations

from typing import Any

from weather_agent import memory
from weather_agent.errors import tool_error


def save_preference(key: str, value: str) -> dict[str, Any]:
    """保存一条用户偏好（跨会话生效）。

    Args:
        key: 偏好的名字，如 region_focus、language、unit
        value: 偏好的内容，如 "eastern_pacific"、"中文"、"摄氏度"
    """
    try:
        result = memory.remember(key, value)
        return {
            **result,
            "note": "已写入长期记忆，下次启动会注入到系统提示",
        }
    except Exception as exc:
        return tool_error(exc, "保存用户偏好")


def recall_preferences(key: str | None = None) -> dict[str, Any]:
    """读取已保存的用户偏好。

    Args:
        key: 指定偏好名；不传则返回全部
    """
    try:
        if key:
            row = memory.recall(key)
            return {"key": key, "value": row["value"] if row else None, "found": bool(row)}
        rows = memory.recall() or []
        return {"count": len(rows), "preferences": rows}
    except Exception as exc:
        return tool_error(exc, "读取用户偏好")


def forget_preference(key: str) -> dict[str, Any]:
    """删除一条用户偏好。

    Args:
        key: 要删除的偏好名
    """
    try:
        return memory.forget(key)
    except Exception as exc:
        return tool_error(exc, "删除用户偏好")


save_preference.tool_info = {
    "tool_name": "save_preference",
    "tool_title": "记住用户偏好",
    "tool_description": "把用户明确表达的长期偏好写入跨会话记忆（如关注的区域、分型、语言、单位）。只在用户明确表达偏好时调用，不要把当次查询的数值写进去。",
    "tool_params": [
        {"name": "key", "description": "偏好名，如 region_focus / language / unit", "type": "string", "required": True},
        {"name": "value", "description": "偏好内容，如 eastern_pacific / 中文 / 摄氏度", "type": "string", "required": True},
    ],
}

recall_preferences.tool_info = {
    "tool_name": "recall_preferences",
    "tool_title": "读取用户偏好",
    "tool_description": "读取已保存的用户偏好；系统提示里已注入的部分不必重复查询。",
    "tool_params": [
        {"name": "key", "description": "指定偏好名；不传返回全部", "type": "string", "required": False},
    ],
}

forget_preference.tool_info = {
    "tool_name": "forget_preference",
    "tool_title": "删除用户偏好",
    "tool_description": "删除一条已保存的用户偏好。用户明确说\"不要记这个\"时调用。",
    "tool_params": [
        {"name": "key", "description": "要删除的偏好名", "type": "string", "required": True},
    ],
}
