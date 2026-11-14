#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 检索评估：recall@k、MRR、nDCG@k、领域外误召回率。

为什么需要独立评估：
端到端问答只能说明"最终答案对不对"，答对了可能靠模型常识；
答错了也分不清是"没召回"还是"召回了但没用"。这里直接测检索本身。

数据：
- retrieval_golden.json：30 条领域内问题（标注应召回片段）+ 10 条领域外问题
- 片段 id 与 knowledge.db 对应，语料重建后 id 可能变化，需要同步维护标注

用法：
    python weather_agent/checks/eval_retrieval.py             # 关键词 + 向量 + 混合
    python weather_agent/checks/eval_retrieval.py --no-vector # 只测关键词（CI 无密钥时）
    python weather_agent/checks/eval_retrieval.py --json out.json
    python weather_agent/checks/eval_retrieval.py --gate      # 指标低于阈值时退出码 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "retrieval_golden.json"

# 阈值按首次实测设定，留出小幅波动空间；语料或 embedding 模型变化后需要重标
GATES = {
    "keyword": {"recall": 0.65, "mrr": 0.65, "ood_fp": 0.20},
    "vector": {"recall": 0.90, "mrr": 0.85, "ood_fp": 0.10},
    "hybrid": {"recall": 0.90, "mrr": 0.85, "ood_fp": 0.10},
}


def load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def mrr(ranked: list[str], relevant: list[str]) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def _keyword(query: str, k: int) -> list[str]:
    from weather_agent.knowledge import store

    return [item["id"] for item in store.search(query, k)]


def _vector(query: str, k: int) -> list[str]:
    from weather_agent.knowledge import embeddings, vector_store

    if not embeddings.available():
        raise RuntimeError("EMBEDDING_* 未配置")
    hits = vector_store.search(embeddings.embed_one(query), k, score_threshold=0.42)
    return [item["id"] for item in hits]


def _hybrid(query: str, k: int) -> list[str]:
    from weather_agent.tools.knowledge import _hybrid as hybrid

    return [item["id"] for item in hybrid(query, k)[0]]


MODES = {"keyword": _keyword, "vector": _vector, "hybrid": _hybrid}


def evaluate(mode: str, golden: dict[str, Any], k: int, verbose: bool = False) -> dict[str, Any]:
    fn = MODES[mode]
    rows: list[dict[str, Any]] = []
    for case in golden["queries"]:
        try:
            ranked = fn(case["q"], k)
        except Exception as exc:
            return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
        row = {
            "q": case["q"],
            "relevant": case["relevant"],
            "ranked": ranked,
            "recall": recall_at_k(ranked, case["relevant"], k),
            "mrr": mrr(ranked, case["relevant"]),
            "ndcg": ndcg_at_k(ranked, case["relevant"], k),
        }
        rows.append(row)
        if verbose and row["recall"] < 1.0:
            print(f"  [MISS] {case['q']} -> {ranked} (期望 {case['relevant']})")

    ood_hits = 0
    for query in golden.get("ood", []):
        try:
            ranked = fn(query, k)
        except Exception as exc:
            return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
        if ranked:
            ood_hits += 1
            if verbose:
                print(f"  [OOD ] {query} -> {ranked}")

    n = len(rows) or 1
    return {
        "mode": mode,
        "k": k,
        "queries": len(rows),
        "recall": round(sum(row["recall"] for row in rows) / n, 3),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 3),
        "ndcg": round(sum(row["ndcg"] for row in rows) / n, 3),
        "hit_rate": round(sum(1 for row in rows if row["mrr"] > 0) / n, 3),
        "ood_cases": len(golden.get("ood", [])),
        "ood_fp_rate": round(ood_hits / max(1, len(golden.get("ood", []))), 3),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评估")
    parser.add_argument("--no-vector", action="store_true", help="跳过需要 embedding 的模式")
    parser.add_argument("--json", help="把完整结果写入指定文件")
    parser.add_argument("--gate", action="store_true", help="低于阈值时返回非零退出码")
    parser.add_argument("--verbose", action="store_true", help="打印未完全召回的问题")
    args = parser.parse_args(argv)

    golden = load_golden()
    k = int(golden.get("top_k", 4))
    modes = ["keyword"] if args.no_vector else ["keyword", "vector", "hybrid"]

    results: dict[str, Any] = {"k": k, "results": {}}
    failures: list[str] = []
    print(f"{'模式':<8}{'recall@'+str(k):<12}{'MRR':<8}{'nDCG@'+str(k):<10}{'命中率':<8}{'领域外误召回':<10}")
    for mode in modes:
        result = evaluate(mode, golden, k, verbose=args.verbose)
        results["results"][mode] = result
        if "error" in result:
            print(f"{mode:<8}跳过（{result['error']}）")
            continue
        print(f"{mode:<8}{result['recall']:<12}{result['mrr']:<8}{result['ndcg']:<10}"
              f"{result['hit_rate']:<8}{result['ood_fp_rate']:<10}")
        gate = GATES.get(mode)
        if args.gate and gate:
            if result["recall"] < gate["recall"]:
                failures.append(f"{mode} recall {result['recall']} < {gate['recall']}")
            if result["mrr"] < gate["mrr"]:
                failures.append(f"{mode} MRR {result['mrr']} < {gate['mrr']}")
            if result["ood_fp_rate"] > gate["ood_fp"]:
                failures.append(f"{mode} OOD {result['ood_fp_rate']} > {gate['ood_fp']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"结果已写入 {args.json}")

    if failures:
        print("门禁未通过：")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
