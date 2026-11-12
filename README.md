# enso-analyst

![CI](https://github.com/walkerman022/enso-analyst/actions/workflows/weather-agent.yml/badge.svg)

A tool-augmented LLM agent for ENSO monitoring, analysis and forecasting.

The agent answers questions such as *"is this an El Niño year?"*, *"how is ONI defined?"* or
*"what do the next months look like?"* by calling data tools, retrieving a curated knowledge base,
and running a trained ENSO model. Its distinguishing feature is an **evidence ledger**: every number
an answer cites is registered before the tool returns, then verified after the answer is generated.

## Why an evidence ledger

Prompt-level instructions ("only use tool results") are a constraint, not a guarantee. The ledger makes
the check deterministic:

1. Tools register each numeric value with its source and the period it belongs to (e.g. `2.46 / ASO 2026`).
2. After generation, every number in the answer is matched against the ledger — exact, by object, or derived.
3. Knowledge and web citations are checked against the ids actually retrieved in this run.
4. Unverified values and mismatched periods are flagged (`annotate`) or blocked (`--strict`).

Covered error types: fabricated values, period mismatch, fabricated citations.
Not covered: qualitative claims and undefined comparison statements.

## Capabilities

- **Monitoring / statistics** — ONI, SOI, Niño 1+2/3/4, extremes, historical events, analogues
- **Knowledge retrieval** — SQLite + Qdrant hybrid search (Chinese bigram + 1024-d vectors, RRF fusion),
  with optional web fallback (Bing HTML) when the local corpus has no hit and `--web` is on
- **Forecasting** — a trained physics-constrained ENSO model served in a separate Python environment;
  outputs are marked as forecasts, with `issued_from` / `data_through` / `model_id`
- **Council mode** — three role-scoped agents (ocean / atmosphere / history) plus a tool-less chief
  that must report disagreements rather than averaging them away

17 business tools in total.

## Evaluation

| Layer | What it checks | Size |
|---|---|---|
| Code-level | ledger binding, compaction, isolation, budget, cache, prediction, web fallback | 148 assertions |
| End-to-end | answer correctness (data / knowledge / council / prediction) | 26 cases |
| Tool choice | expected tool appears in the call sequence | 20 labels |
| Stability | same question, three runs, core values agree | 5 questions × 3 |

The 148 offline assertions run in CI (`.github/workflows/weather-agent.yml`) and are compared against
`weather_agent/checks/baseline.json`; a drop in passing count fails the build.

```bash
python weather_agent/checks/ci_check.py          # run and compare with baseline
python weather_agent/checks/ci_check.py --update # refresh baseline
python weather_agent/checks/eval_set.py          # end-to-end (requires API key)
```

## Quick start

```bash
pip install -r requirements.txt

export DEEPSEEK_API_KEY=...          # chat model
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export EMBEDDING_API_KEY=...         # Qwen embedding endpoint (DashScope compatible)
export EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export EMBEDDING_MODEL=qwen3.7-text-embedding-flash

python -m weather_agent.knowledge.build_index   # build once
python -m weather_agent.cli "现在算不算厄尔尼诺？"
python -m weather_agent.cli --session demo "那未来三个月呢？"
python -m weather_agent.cli --web "2026 年厄尔尼诺对秘鲁渔业有什么报道？"
python -m weather_agent.cli --strict "ONI 最新值是多少？"   # block instead of annotate
python -m weather_agent.council "这次厄尔尼诺和 1997 年比怎么样？"
```

Forecasting requires the model environment configured in `weather_agent/models.json`
(separate interpreter with PyTorch); the Agent process does not import torch.

## Repository layout

```
weather_agent/
  agent.py            assembly: instructions, tools, budget, guardrails
  prompts.py          17 rules, including forecast/observation separation
  evidence.py         ledger (value + source + field + object), scoped per role
  number_check.py     five-level verification + citation checks
  cache.py            TTL text cache for upstream data
  knowledge/          corpus, chunking, hybrid retrieval, citation registry
  tools/              data, statistics, plotting, retrieval, prediction
  checks/             code-level assertions, end-to-end evals, CI gate, baseline
  council.py          three experts + tool-less chief
  web_search.py       optional web fallback with snippet sanitisation
LightAgent/           agent framework (Apache-2.0, see README.zh-CN.md)
```

## Data and limits

- ONI/SOI from NOAA CPC; regional indices from NOAA PSL; knowledge base from NOAA, CPC and CMA pages
  plus 24 hand-written cards.
- Forecast checkpoint was trained on data through 2026-02; real-time forecasting is therefore not
  available. The agent reports `data_through` and refuses to present forecasts as observations.
- Multi-user operation is not supported: preferences are global, sessions have no authentication,
  and both SQLite stores assume a single writer.
- No structured output yet: quantitative verification relies on text extraction, so clauses with
  several periods or purely qualitative comparisons are only partially covered.

## License

The agent code in `weather_agent/` is released under the MIT License.
It builds on LightAgent v0.11.1 (Apache-2.0); the framework license is retained in `LICENSE`.
