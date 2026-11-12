#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""带提示缓存的 Agent 包装：命中直接返回，未命中才调模型。

为什么要包一层而不是改框架：
缓存是业务策略（TTL 多长、什么情况下不能缓存），不该塞进 Agent 内核；
包一层也能保证"框架升级不影响缓存逻辑"。

不缓存的情况（都会显式跳过）：
- 带 session_id 的续聊：历史在框架内部，无法稳定算出键；
- 调用了写操作工具（save_preference / forget_preference）：有副作用，重放会造成重复写入；
- 回答以 [LA- 开头：错误结果不缓存，避免把故障固化。
"""

from __future__ import annotations

import functools
from typing import Any

from weather_agent import memory, prompt_cache
from weather_agent.tool_recorder import ToolRecorder
from weather_agent.agent import build_agent
from weather_agent.knowledge import store as knowledge_store
from weather_agent.tools import ALL_TOOLS

SIDE_EFFECT_TOOLS = {"save_preference", "forget_preference", "plot_enso_trend", "plot_enso_regions"}


class CachedAgent:
    """对 LightAgent 的薄包装，提供"同提示直接返回"的能力。"""

    def __init__(self, *, ttl: int = prompt_cache.DEFAULT_TTL, enabled: bool = True, **build_kwargs: Any):
        self.recorder = ToolRecorder()
        tools = build_kwargs.pop("tools", None) or ALL_TOOLS
        self.agent = build_agent(tools=self.recorder.wrap(tools), **build_kwargs)
        self.ttl = int(ttl)
        self.enabled = bool(enabled)
        self.last_hit = False

    # 透传常用属性，避免调用方到处写 .agent
    def __getattr__(self, item: str) -> Any:
        return getattr(self.agent, item)

    def _key(self, question: str, history: list[dict[str, str]] | None) -> str:
        messages = list(history or []) + [{"role": "user", "content": question}]
        return prompt_cache.signature(
            model=str(getattr(self.agent, "model", "")),
            instructions=str(getattr(self.agent, "instructions", "")),
            messages=messages,
            tools_schema=list(self.agent.tool_registry.openai_function_schemas),
            environment={"prefs": memory.version(), "kb": knowledge_store.version()},
        )

    def run(
            self,
            question: str,
            history: list[dict[str, str]] | None = None,
            session_id: str | None = None,
    ) -> str:
        self.last_hit = False
        if not self.enabled or session_id:
            return self.agent.run(question, history=history, session_id=session_id)

        key = self._key(question, history)
        cached = prompt_cache.get(key)
        if cached is not None:
            self.last_hit = True
            return cached

        self.recorder.reset()
        text = self.agent.run(question, history=history)
        text = text if isinstance(text, str) else str(text)

        has_side_effect = any(name in SIDE_EFFECT_TOOLS for name in self.recorder.names())
        if not has_side_effect and not text.startswith("[LA-"):
            prompt_cache.put(key, text, ttl=self.ttl, tags=["answer"])
        return text

    def stats(self) -> dict[str, Any]:
        return prompt_cache.stats()


def build_cached_agent(**kwargs: Any) -> CachedAgent:
    return CachedAgent(**kwargs)
