#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对话摘要器：给上下文压缩器用，把"即将被丢弃的旧消息"压成一段简短说明。

为什么摘要里**禁止出现阿拉伯数字**：
本项目的输出护栏靠"答案里的数字能在账本里找到"来判定幻觉。摘要是一次模型改写，
如果它把"ONI 2.46"写成"约 2.5"，模型后面引用摘要里的数字，回查就会判成"无出处"。
所以摘要只保留"意图、前提、时间范围"，数值一律不带——要数据就让模型重新查。

三重保护：
1. 提示词明确禁止数字；
2. 生成后再做一次清洗（删掉阿拉伯数字与百分号）；
3. 模型调用失败 → 用确定性回退摘要（截取用户原话、同样去数字）。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from weather_agent.credentials import load_credentials

# 匹配阿拉伯数字（含小数）和百分号。中文数字（"三"）不处理：
# 护栏的正则只认阿拉伯数字，中文数字本来就不参与回查。
_NUMBER_CLEANER = re.compile(r"\d+(?:\.\d+)?\s*%?")

SUMMARY_SYSTEM = """你在帮一个对话系统做上下文压缩。把你收到的旧对话压成一段不超过 150 字的说明。

硬性要求：
1. 禁止出现任何阿拉伯数字、小数、百分号。需要指代数据时写"先前查询的 ONI 结果"这类说法。
2. 只写三件事：用户在做什么、已经确认的前提、涉及的时间范围（用"早前""上一轮"这类相对说法）。
3. 不要复述助手的具体结论，不要评价，不要补充新信息。

4. 如果输入里带有"已有摘要"，必须把其中的意图、前提、范围合并进新摘要，不能丢掉。
   连续压缩时，旧摘要就是唯一还留着的早期信息。

输出格式（三行，不加标题符号）：
意图：……
前提：……
范围：……"""


def strip_numbers(text: str) -> str:
    """删掉阿拉伯数字，用于摘要清洗。"""
    return _NUMBER_CLEANER.sub("", text or "")


def fallback_summary(messages: list[dict[str, Any]], limit: int = 240) -> str:
    """确定性回退：截取用户原话，去掉数字。

    不如模型摘要连贯，但满足两条硬约束：一定不含数字、一定不会调用失败。
    """
    lines: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = strip_numbers(str(message.get("content", ""))).strip()
        if content:
            lines.append(content[:24])
    if not lines:
        return "早前对话已压缩（内容为空）。"
    return "早前对话已压缩，用户先后问过：" + "；".join(lines[-4:]) + "。"


def make_summarizer(
        *,
        model: str | None = None,
        timeout: float = 60.0,
        max_chars: int = 600,
) -> Callable[[list[dict[str, Any]]], str]:
    """返回一个可直接传给 ContextCompactor 的摘要函数。"""
    credentials = load_credentials()
    from openai import OpenAI

    client = OpenAI(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=timeout,
        max_retries=0,
    )
    target_model = model or credentials["model"]
    cache: dict[str, str] = {}   # 同一批被丢弃的消息只摘要一次

    def summarize(messages: list[dict[str, Any]]) -> str:
        fingerprint = hashlib.sha1(
            "\n".join(str(item.get("content", "")) for item in messages).encode("utf-8")
        ).hexdigest()
        if fingerprint in cache:
            return cache[fingerprint]
        # 只把"用户说了什么"喂给模型，助手回复太长且容易带数字
        previous = [
            item for item in messages
            if item.get("role") == "system" and (item.get("metadata") or {}).get("compacted")
        ]
        history = "\n".join(
            f"{item.get('role')}: {str(item.get('content', ''))[:400]}"
            for item in messages
            if item.get("role") in ("user", "assistant")
        )
        if previous:
            history = (
                "已有摘要（必须合并进新摘要，不得丢弃）：\n"
                + str(previous[-1].get("content", ""))[:800]
                + "\n\n新增对话：\n"
                + history
            )
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": history or "（无内容）"},
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            # 摘要失败不能影响主流程：退回确定性摘要
            return fallback_summary(messages)

        cleaned = strip_numbers(text).strip()
        if len(cleaned) < 10:
            cleaned = fallback_summary(messages)
        cleaned = cleaned[:max_chars]
        if len(cache) < 8:
            cache[fingerprint] = cleaned
        return cleaned

    return summarize
