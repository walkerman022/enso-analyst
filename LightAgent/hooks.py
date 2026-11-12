#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runtime hook primitives for LightAgent lifecycle extensions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Any, Callable


HOOK_CONTINUE = "continue"
HOOK_REPLACE = "replace"
HOOK_BLOCK = "block"
HOOK_RETRY = "retry"
HOOK_FALLBACK = "fallback"
HOOK_METADATA = "metadata"


@dataclass
class HookContext:
    """Context passed to runtime hooks."""
    """纯数据容器，'当前时刻+当前数据+这次运行的身份'打包递给钩子"""
    phase: str                                                      # 当前是哪个时刻
    payload: dict[str, Any] = field(default_factory=dict)           # 这一时刻的数据
    trace_id: str | None = None                                     # 本次运行的追踪号
    parent_trace_id: str | None = None                              # 上游运行的追踪号
    run_id: str | None = None                                       # 本次运行的id
    run_group_id: str | None = None                                 # 运行组的id
    user_id: str | None = None                                      # 谁在问
    agent_name: str | None = None                                   # 哪个agent
    flow_id: str | None = None                                      # 编排流程id
    step_name: str | None = None                                    # 流程步骤名
    metadata: dict[str, Any] = field(default_factory=dict)          #


@dataclass
class HookDecision:
    """Decision returned by a runtime hook."""
    """HookManager.run使用，告诉框架接下来干什么"""
    action: str = HOOK_CONTINUE
    payload: dict[str, Any] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # 放行
    @classmethod
    def continue_(cls, *, metadata: dict[str, Any] | None = None) -> "HookDecision":
        return cls(action=HOOK_CONTINUE, metadata=metadata or {})

    # 改写payload，放行
    @classmethod
    def replace(cls, payload: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> "HookDecision":
        return cls(action=HOOK_REPLACE, payload=payload, metadata=metadata or {})

    # 拦截
    @classmethod
    def block(cls, reason: str | None = None, *, metadata: dict[str, Any] | None = None) -> "HookDecision":
        return cls(action=HOOK_BLOCK, reason=reason, metadata=metadata or {})

    # 重试
    @classmethod
    def retry(cls, reason: str | None = None, *, metadata: dict[str, Any] | None = None) -> "HookDecision":
        return cls(action=HOOK_RETRY, reason=reason, metadata=metadata or {})

    # 降级
    @classmethod
    def fallback(cls, reason: str | None = None, *, metadata: dict[str, Any] | None = None) -> "HookDecision":
        return cls(action=HOOK_FALLBACK, reason=reason, metadata=metadata or {})


@dataclass(frozen=True)
class PolicyHook:
    """Explicit hook policy for failures that must block protected operations."""
    """对钩子进行包装，给裸函数加上三样保险"""

    handler: Callable[..., Any] | Any
    phases: set[str] | frozenset[str] | None = None
    failure_mode: str = "block"                        # 出错时拦截还是放行
    timeout: float | None = None
    name: str | None = None

    # @dataclass 的对象创建完后会自动调用这个方法，用来做校验和补充
    def __post_init__(self) -> None:
        # handler 不能是 None
        if self.handler is None:
            raise ValueError("PolicyHook handler is required")
        # failure_mode 只能是 "block" 或 "continue"
        if self.failure_mode not in {"block", "continue"}:
            raise ValueError("PolicyHook failure_mode must be 'block' or 'continue'")
        # timeout 必须是大于 0 的数字（且不能是 bool）
        if self.timeout is not None:
            if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
                raise ValueError("PolicyHook timeout must be a number greater than 0")
        # 把 phases 转成 frozenset
        if self.phases is not None:
            object.__setattr__(self, "phases", frozenset(str(phase) for phase in self.phases))


class HookManager:
    """Run hooks in list order while isolating observability hook failures."""
    """按顺序跑完所有钩子，收集它们的影响，返回最终决定"""

    def __init__(self, hooks: list[Callable[..., Any] | Any | PolicyHook] | None = None):
        self.hooks = list(hooks or [])

    def run(self, context: HookContext) -> HookDecision:
        payload = dict(context.payload or {})                   # 当前payload，会被改写。payload 复制一份，后面改它不会影响调用方传进来的那个字典。
        hook_events: list[dict[str, Any]] = []                  # 每个钩子的动作，事后写 trace
        metadata: dict[str, Any] = {}                           # 累积 metadata
        pending_action: HookDecision | None = None              # retry/fallback：记下来，最后再返回

        for hook in self.hooks:
            policy = hook if isinstance(hook, PolicyHook) else None     # 判断是包装盒吗
            handler = policy.handler if policy else hook                # 真正的函数
            if policy and policy.phases is not None and context.phase not in policy.phases:
                continue                                                # 阶段不匹配，跳过
            """链式兜底"""
            """第一级：PolicyHook.name"""
            """只在钩子被 PolicyHook 包装时才有值。没有包装（裸函数）时，这段返回 None"""
            """第二级：handler.__name__"""
            """第三级：handler.__class__.__name__"""
            hook_name = (policy.name if policy else None) or getattr(
                handler,
                "__name__",
                handler.__class__.__name__,
            )
            try:
                # _invoke_hook负责调用
                raw = self._invoke_hook(handler, context, policy.timeout if policy else None)
                decision = self._normalize(raw)     # _normalize负责把各种返回值统一成HookDecision
            except Exception as exc:
                failure_mode = policy.failure_mode if policy else "continue"
                error_event = {
                    "phase": context.phase,
                    "hook": hook_name,
                    "action": "error",
                    "error": str(exc),
                    "error_type": "timeout" if isinstance(exc, TimeoutError) else "exception",
                    "failure_mode": failure_mode,
                }
                hook_events.append(error_event)
                if failure_mode == "block":
                    return HookDecision.block(
                        f"Policy hook `{hook_name}` failed closed: {exc}",
                        metadata={
                            **metadata,
                            "policy_hook": hook_name,
                            "failure_mode": failure_mode,
                            "hook_events": hook_events,
                        },
                    )       # fail-closed：立刻拦截
                continue    # fail-open：记一笔，继续

            if decision.metadata:
                metadata.update(decision.metadata)          # metadata 累积，不覆盖.每个钩子的 metadata 汇进同一个字典。

            if decision.action == HOOK_REPLACE:             # 改写 payload，继续
                payload = dict(decision.payload or {})
                context.payload = payload
                hook_events.append({
                    "phase": context.phase,
                    "hook": hook_name,
                    "action": HOOK_REPLACE,
                })
                continue

            if decision.action == HOOK_METADATA:            # 什么都不做，只记一笔
                hook_events.append({
                    "phase": context.phase,
                    "hook": hook_name,
                    "action": HOOK_METADATA,
                })
                continue

            if decision.action in (HOOK_RETRY, HOOK_FALLBACK):
                # retry / fallback 全项目没有消费方（没有 if action == retry 的分支），
                # 原先会直接中断后续钩子，等于“设了 retry 却静默丢掉后面的钩子”。
                # 这里改为记一笔继续跑，最后仍把该动作返回给调用方。
                pending_action = decision
                hook_events.append({
                    "phase": context.phase,
                    "hook": hook_name,
                    "action": decision.action,
                    "reason": decision.reason,
                })
                continue

            if decision.action != HOOK_CONTINUE:            # 非continue,立即返回
                decision.payload = payload
                decision.metadata = {**metadata, **decision.metadata, "hook_events": hook_events}
                hook_events.append({
                    "phase": context.phase,
                    "hook": hook_name,
                    "action": decision.action,
                    "reason": decision.reason,
                })
                decision.metadata["hook_events"] = hook_events
                return decision

        """即使没有钩子修改任何东西，也会返回一个 continue 的 decision，把累积的 payload 和 metadata 带回去。"""
        if pending_action is not None:
            pending_action.payload = payload
            pending_action.metadata = {
                **metadata,
                **pending_action.metadata,
                "hook_events": hook_events,
            }
            return pending_action

        return HookDecision(
            action=HOOK_CONTINUE,
            payload=payload,                                # 可能被前面的 replace 改过
            metadata={**metadata, "hook_events": hook_events} if hook_events else metadata,
        )

    """调用钩子"""
    """用三种方式尝试调用钩子，支持函数和对象两种写法"""
    @staticmethod
    def _call_hook(hook: Callable[..., Any] | Any, context: HookContext) -> Any:
        """如果context.phase是before_run，就取hook.before_run"""
        method = getattr(hook, context.phase, None)
        if callable(method):
            return method(context)
        """普通函数、lambda、functools.partial 都走这条"""
        if callable(hook):
            return hook(context)
        return None

    '''没设超时就直接调；设了就丢进线程池跑，超时抛异常'''
    @classmethod
    def _invoke_hook(
            cls,
            hook: Callable[..., Any] | Any,
            context: HookContext,
            timeout: float | None,
    ) -> Any:
        # 没设超时直接跑，绝大数钩子走这条路
        if timeout is None:
            return cls._call_and_resolve(hook, context)

        isolated_context = HookContext(
            phase=context.phase,
            payload=dict(context.payload),               # 浅拷贝
            trace_id=context.trace_id,
            parent_trace_id=context.parent_trace_id,
            run_id=context.run_id,
            run_group_id=context.run_group_id,
            user_id=context.user_id,
            agent_name=context.agent_name,
            flow_id=context.flow_id,
            step_name=context.step_name,
            metadata=dict(context.metadata),            # 浅拷贝
        )
        """钩子拿到的是副本,所以在设了超时的钩子里改的是副本"""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(cls._call_and_resolve, hook, isolated_context)
        try:
            return future.result(timeout=timeout)       # 最多等 timeout 秒
        except concurrent.futures.TimeoutError as exc:
            future.cancel()                             # 尝试取消
            raise TimeoutError(f"hook timed out after {timeout:g}s") from exc       # 转成内置 TimeoutError
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        # "另开一个线程 + 复制一份数据"的代价：
        # 1.改不了主流程的 payload           原因：拿到的是副本
        # 2.超时后线程不会真停                原因：cancel() 只能取消"还没开始"的任务
        # 3.每次调用都新建线程池              原因： 高频钩子会有额外开销

    # 异步————————————————————————————————————————————————————————————————————————————————————————
    '''调用钩子，如果返回的是协程就帮你跑完'''
    @classmethod
    def _call_and_resolve(cls, hook: Callable[..., Any] | Any, context: HookContext) -> Any:
        raw = cls._call_hook(hook, context)
        if inspect.isawaitable(raw):
            return cls._run_awaitable(raw)
        return raw

    '''把协程变成同步结果，分两种情况'''
    @staticmethod
    def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
        try:
            asyncio.get_running_loop()                      # 当前线程有没有正在运行的事件循环？
        except RuntimeError:
            return asyncio.run(awaitable)                   # 路 A：没有，直接跑

        # 路 B：有循环，另开线程跑
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, awaitable)
            return future.result()
    # ——————————————————————————————————————————————————————————————————————————————————————————————

    # 返回钩子的任意值，都输出统一的HookDecision
    @staticmethod
    def _normalize(raw: Any) -> HookDecision:
        if raw is None:
            return HookDecision.continue_()
        if isinstance(raw, HookDecision):
            return raw
        if isinstance(raw, bool):
            return HookDecision.continue_() if raw else HookDecision.block("hook returned False")
        if isinstance(raw, str):
            return HookDecision.block(raw)
        if isinstance(raw, dict):
            if raw.get("allowed") is False:
                return HookDecision.block(raw.get("reason"), metadata=raw.get("metadata"))
            action = raw.get("action")
            if action:
                return HookDecision(
                    action=str(action),
                    payload=raw.get("payload"),
                    reason=raw.get("reason"),
                    metadata=raw.get("metadata") or {},
                )
            if "payload" in raw:
                return HookDecision.replace(raw["payload"], metadata=raw.get("metadata"))
            return HookDecision.continue_(metadata=raw.get("metadata"))
        return HookDecision.continue_()
