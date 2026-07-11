# Eval history

`runs.jsonl` is an append-only record of every `run_eval` and ablation
configuration executed against this repo. One JSON line per run:

```json
{
  "timestamp": "2026-07-03T12:34:56+00:00",
  "git_commit": "abc1234",
  "strategy": "hybrid",
  "corrective": false,
  "n_cases": 30,
  "passed": true,
  "mean_context_recall": 0.87,
  "mean_context_precision": 0.72,
  "mean_faithfulness": 0.91,
  "mean_correctness": 0.83,
  "mean_answer_relevancy": 0.89,
  "latency_p50_ms": 1832,
  "latency_p95_ms": 4210,
  "total_cost_usd": 0.0237
}
```

Every commit that changes retrieval or generation should be reflected in a
new line here. Trends over time - quality drift, latency regressions,
cost blowups - are plottable directly from this file.

Never edit an existing line. Prunable later if the file grows unwieldy.

Load in Python:

```python
from rag_harness.evaluation.history import load_history
entries = load_history()  # list[HistoryEntry]
```
