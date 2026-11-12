#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提示缓存实测：命中率与一致性。

做三件事：
1. 同一问题问两次，第二次必须命中，且回答逐字一致、模型调用为 0；
2. 换问题必须未命中；
3. 改偏好后（键里带版本号）同一问题必须重新 miss——验证"环境变化自动失效"。

    python weather_agent/checks/eval_cache.py
    python weather_agent/checks/eval_cache.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import weather_agent  # noqa: F401

from weather_agent import memory, metrics, prompt_cache  # noqa: E402
from weather_agent.cached_agent import build_cached_agent  # noqa: E402

QUESTIONS = [
    "最近一期 ONI 是多少？",
    "现在算不算厄尔尼诺？",
    "ONI 是怎么定义的？",
    "东部型和中部型有什么区别？",
    "1997 年那次厄尔尼诺峰值多少？",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提示缓存实测")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    prompt_cache.invalidate()          # 从干净状态开始
    agent = build_cached_agent()

    first_texts: dict[str, str] = {}
    hits = 0
    consistent = 0
    zero_calls = 0

    for index, question in enumerate(QUESTIONS, start=1):
        first = agent.run(question)
        # 注意：框架的 model_calls 是"本次 run"的计数，命中缓存时不会归零，
        # 所以判断"有没有真的调模型"要看累计 token 的增量——命中时增量为 0。
        tokens_before = metrics.snapshot(agent.agent)["tokens"]
        second = agent.run(question)
        tokens_delta = metrics.snapshot(agent.agent)["tokens"] - tokens_before

        hit = bool(agent.last_hit)
        same = first == second
        hits += int(hit)
        consistent += int(same)
        no_model_call = tokens_delta == 0
        zero_calls += int(no_model_call)
        first_texts[question] = first

        print(f"[{index}/{len(QUESTIONS)}] 命中={'是' if hit else '否'} "
              f"一致={'是' if same else '否'} 第二次 token 增量={tokens_delta:.0f}")
        if args.verbose:
            print(f"        {question}\n        {first[:100]}")

    # 换问题应未命中
    agent.run("拉尼娜的定义是什么？")
    new_question_missed = not agent.last_hit
    print(f"换新问题未命中：{'是' if new_question_missed else '否'}")

    # 改偏好后，同一问题应重新 miss
    before_version = memory.version()
    memory.remember("_cache_probe", "临时探针")
    try:
        agent.run(QUESTIONS[0])
        env_change_missed = not agent.last_hit
        print(f"偏好变化后同问题重新未命中：{'是' if env_change_missed else '否'}")
    finally:
        memory.forget("_cache_probe")

    stats = prompt_cache.stats()
    print(f"\n缓存条目 {stats['entries']}，命中 {stats['hits']}，未命中 {stats['misses']}，"
          f"命中率 {stats['hit_rate']:.1%}")
    print(f"一致 {consistent}/{len(QUESTIONS)}，零调用 {zero_calls}/{len(QUESTIONS)}")

    ok = hits == len(QUESTIONS) and consistent == len(QUESTIONS) and zero_calls == len(QUESTIONS) \
        and new_question_missed and env_change_missed
    print("结论：" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
