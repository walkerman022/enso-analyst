#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成忠实度评估：被引用的陈述句是否真被所引片段支持。

样本：`faithfulness_cases.json`，8 条真实问答（含引用），离线判分，不调模型。
指标：
- faithfulness：带引用的陈述句中，被所引片段支持的比例
- citation_coverage：全部陈述句中，带引用的比例（低说明模型少标出处）

判分方式见 faithfulness.py：词面重合 ≥0.35、数字命中、或前 3 个高 IDF 术语命中 2 个。
门禁：faithfulness ≥ 0.70，citation_coverage ≥ 0.35。

局限：样本是固定快照，模型或语料变化后需要重新采集；同义改写仍可能误判。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

from weather_agent.checks.faithfulness import score_answer
from weather_agent.knowledge.store import tokenize

HERE = Path(__file__).resolve().parent
CASES = HERE / "faithfulness_cases.json"
DB = HERE.parent / "data" / "knowledge.db"

GATE_FAITHFULNESS = 0.70
GATE_COVERAGE = 0.35


def load_idf() -> dict[str, float]:
    connection = sqlite3.connect(DB)
    try:
        texts = [row[0] for row in connection.execute("SELECT text FROM chunks")]
    finally:
        connection.close()
    total = len(texts)
    df: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            df[token] = df.get(token, 0) + 1
    return {token: math.log((total + 1) / (count + 0.5)) for token, count in df.items()}


def load_chunks(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    connection = sqlite3.connect(DB)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = connection.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
    finally:
        connection.close()
    return {row[0]: row[1] for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成忠实度评估")
    parser.add_argument("--json", help="把逐条结果写入文件")
    parser.add_argument("--gate", action="store_true", help="低于阈值时返回非零退出码")
    parser.add_argument("--verbose", action="store_true", help="打印未支持句子")
    args = parser.parse_args(argv)

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    idf = load_idf()

    total_claims = total_cited = total_supported = 0
    details: list[dict] = []
    for case in cases:
        ids = list(dict.fromkeys(case.get("retrieved", []) + case.get("cited", [])))
        chunks = load_chunks(ids)
        result = score_answer(case["answer"], chunks, idf=idf)
        total_claims += result["claims"]
        total_cited += result["cited_claims"]
        total_supported += result["cited_supported"]
        details.append({"q": case["q"], **result})
        print(f"{case['q'][:16]:18} claims={result['claims']:2} "
              f"cited={result['cited_claims']:2} faith={result['faithfulness']:.2f} "
              f"cov={result['citation_coverage']:.2f}")
        if args.verbose:
            for claim in result["unsupported"]:
                print(f"  [UNSUPPORTED] {claim[:90]}")

    faithfulness = round(total_supported / total_cited, 3) if total_cited else 0.0
    coverage = round(total_cited / total_claims, 3) if total_claims else 0.0
    print(f"总体：faithfulness={faithfulness}（{total_supported}/{total_cited}），"
          f"citation_coverage={coverage}（{total_cited}/{total_claims}）")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"faithfulness": faithfulness, "citation_coverage": coverage, "cases": details},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"结果已写入 {args.json}")

    if args.gate and (faithfulness < GATE_FAITHFULNESS or coverage < GATE_COVERAGE):
        print(f"门禁未通过：faithfulness ≥ {GATE_FAITHFULNESS}、coverage ≥ {GATE_COVERAGE}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
