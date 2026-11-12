#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行入口。

用法（在项目根 `D:\\AgentRoom\\LightAgent-0.11.1\\LightAgent-0.11.1` 下运行）：
    python -m weather_agent.cli                     # 交互模式
    python -m weather_agent.cli "最近 ONI 多少"      # 单次提问
    python -m weather_agent.cli --strict "..."      # 数值无出处时直接拦截
    python -m weather_agent.cli --session demo1     # 跨进程保留对话

交互模式下输入 exit / quit 退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from weather_agent import evidence, metrics
from weather_agent.cached_agent import build_cached_agent


def _answer(agent, query: str, history: list[dict[str, str]], session_id: str | None = None) -> str:
    # 用了 session_id 就交给框架从会话记录里投影历史，不要再手动维护 history
    before = metrics.snapshot(agent)
    started = time.perf_counter()
    reply = agent.run(query, history=None if session_id else history, session_id=session_id)
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = reply if isinstance(reply, str) else str(reply)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": text})

    # 记录运行度量：失败不影响回答
    try:
        from weather_agent.number_check import verify

        checked = verify(text, scope="default")
        metrics.record(
            mode="session" if session_id else "single",
            question=query,
            before=before,
            after=metrics.snapshot(agent),
            latency_ms=latency_ms,
            guardrail_issues=len(checked.unverified) + len(checked.mismatched) + len(checked.bad_citations),
            citations=len(checked.citations),
            success=not text.startswith("[LA-"),
            cache_hit=bool(getattr(agent, "last_hit", False)),
        )
    except Exception:
        pass
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ENSO 监测助手（CLI）")
    parser.add_argument("query", nargs="*", help="要问的问题；不给则进入交互模式")
    parser.add_argument("-i", "--interactive", action="store_true", help="连续问答模式")
    parser.add_argument("--strict", action="store_true", help="数值无出处时直接拦截")
    parser.add_argument("--session", help="会话名，用于跨进程保留对话（如 demo1）")
    parser.add_argument("--no-cache", action="store_true", help="关闭提示缓存（默认开启，TTL 10 分钟）")
    parser.add_argument("--web", action="store_true", help="开启联网搜索（默认关闭）")
    parser.add_argument("--debug", action="store_true", help="打印框架调试日志")
    args = parser.parse_args(argv)
    if args.web:
        os.environ["WEB_SEARCH_ENABLED"] = "1"

    store = None
    if args.session:
        # 用框架自带的 SQLite 存储，不自己造轮子；库文件放在 weather_agent/data 下
        from LightAgent.session import SqliteSessionStore

        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        store = SqliteSessionStore(data_dir / "sessions.db")

    agent = build_cached_agent(
        guardrail_mode="block" if args.strict else "annotate",
        debug=args.debug,
        session_store=store,
        enabled=not args.no_cache,
    )
    history: list[dict[str, str]] = []
    # 账本按"会话"清空而不是按轮次：多轮追问时常会引用上一轮查过的数值，
    # 每轮清空会把这种合法引用误判成编造。进程启动时清一次即可。
    evidence.reset()

    if args.query and not args.interactive:
        print(_answer(agent, " ".join(args.query), history, args.session))
        return 0

    print("ENSO 监测助手 · 连续问答（输入 exit 退出）")
    while True:
        try:
            query = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break
        try:
            print(f"\n助手> {_answer(agent, query, history, args.session)}")
        except Exception as exc:  # 单轮失败不该带走整个会话
            print(f"\n[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
