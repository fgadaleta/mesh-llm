# MoE Strategy Benchmarks

This document summarizes the offline MoE strategy benchmark suite added in this branch.

Models tested on `studio54.local`:

- `GLM-4.7-Flash-Q4_K_M`
- `Qwen3-Coder-Next-Q4_K_M`

The suite compares four questions:

1. Which ranking source best matches full `llama-moe-analyze`?
2. How much does ranking quality matter more than grouping shape?
3. Can a short `micro-analyze` replace a full analyze pass?
4. Which live CLI knobs should we expose to test these strategies in mesh runs?

## Bottom Line

- Gold standard: full `llama-moe-analyze`
- Best practical fallback: `micro-analyze` with `--all-layers`
- Best weight-only heuristic in this branch: `heuristic-max`
- Current safe zero-analysis fallback: `sequential`

The strongest result is that `micro-analyze` is dramatically better than any weight-only heuristic on both models, while still costing much less than a full analyze run.

## Ranking Results

### GLM-4.7-Flash-Q4_K_M

| Strategy | Spearman | Recall@24 | Weighted recall@24 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `analyze` | `1.00` | `1.00` | `1.00` | `44.27s` |
| `micro-1p-8t-all-layers` | `1.00` | `1.00` | `1.00` | `17.29s` |
| `heuristic-max` | `0.06` | `0.46` | `0.27` | startup only |
| `sequential` | baseline fallback | baseline fallback | baseline fallback | startup only |

All tested heuristics missed expert `0`, which is unacceptable for this model because full analyze shows expert `0` carries `22.94%` of gate mass.

### Qwen3-Coder-Next-Q4_K_M

| Strategy | Spearman | Recall@256 | Weighted recall@256 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `analyze` | `1.00` | `1.00` | `1.00` | `106.74s` |
| `micro-1p-8t-all-layers` | `0.951` | `0.930` | `0.966` | `32.09s` |
| `micro-4p-32t-all-layers` | `1.00` | `1.00` | `1.00` | `314.95s` |
| `heuristic-max` | `0.020` | `0.516` | `0.741` | startup only |
| `sequential` | baseline fallback | baseline fallback | baseline fallback | startup only |

## Grouping Results

### GLM-4.7-Flash-Q4_K_M

| Grouping strategy | Ranking source | Shared mass | Mean node mass | Imbalance |
| --- | --- | ---: | ---: | ---: |
| `current-analyze` | `analyze` | `52.90%` | `76.45%` | `0.0808%` |
| `snake-analyze-replicated` | `analyze` | `52.90%` | `76.45%` | `0.0338%` |
| `current-sequential` | `sequential` | `51.03%` | `75.52%` | lower risk fallback |
| `snake-heuristic-replicated` | `heuristic-max` | `28.79%` | `64.39%` | `21.27%` |

### Qwen3-Coder-Next-Q4_K_M

| Grouping strategy | Ranking source | Shared mass | Mean node mass | Imbalance |
| --- | --- | ---: | ---: | ---: |
| `current-analyze` | `analyze` | `71.01%` | `85.50%` | `0.0271%` |
| `snake-analyze-replicated` | `analyze` | `71.01%` | `85.50%` | `0.00819%` |
| `current-sequential` | `sequential` | `66.61%` | `83.30%` | lower risk fallback |
| `snake-heuristic-replicated` | `heuristic-max` | `65.55%` | `82.77%` | `0.926%` |

Practical interpretation:

- Ranking quality matters more than grouping shape.
- `snake-draft` is worth testing live, but only when paired with a good ranking source.

## Analysis Cost

Startup cost by strategy:

| Strategy | Work done at startup | Measured cost |
| --- | --- | ---: |
| `bundled / cached analyze` | Local config or CSV read only | file read only |
| `sequential` | GGUF header read only | file read only |
| `heuristic-*` | GGUF tensor scan for router-weight scoring | startup only |
| `micro-analyze` | Short `llama-moe-analyze` run | model-dependent |
| `analyze` | Full `llama-moe-analyze` run | model-dependent |

### Measured analyze timings

Timed on `studio54.local` with:

```bash
/usr/bin/time -lp ./llama-moe-analyze -m MODEL --all-layers --export-ranking /tmp/ranking.csv -n 32 -c 4096 -ngl 99
```

| Model | Full analyze | Micro analyze (`1p/8t/all-layers`) | Notes |
| --- | ---: | ---: | --- |
| `GLM-4.7-Flash-Q4_K_M` | `44.27s` | `17.29s` | micro matched full analyze exactly |
| `Qwen3-Coder-Next-Q4_K_M` | `106.74s` | `32.09s` | micro was already close; larger micro run reached exact match |

## Recommendations

Default behavior should stay conservative for now:

- Keep `auto` as the current stable behavior.
- Prefer `micro-analyze` when we explicitly want a better fallback than sequential.
- Do not make the current weight-only heuristic the default fallback yet.

If we change the default later, this benchmark suggests:

1. `bundled / cached analyze`
2. `micro-analyze --all-layers`
3. `sequential`
4. weight-only heuristics

## Benchmark Commands

Import a small fixed corpus:

```bash
mesh-llm benchmark import-prompts \
  --source mt-bench \
  --limit 8 \
  --max-tokens 256 \
  --output evals/moe/prompts/mt-bench-8.jsonl
```

Run the full offline suite:

```bash
mesh-llm benchmark moe-model-matrix \
  --model /Volumes/External/models/GLM-4.7-Flash-Q4_K_M.gguf \
  --model /Volumes/External/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf \
  --nodes 2 \
  --prompts evals/moe/prompts/mt-bench-8.jsonl \
  --output /tmp/moe-model-matrix.json
```

Run individual slices:

```bash
mesh-llm benchmark moe-heuristic --model /path/to/model.gguf
mesh-llm benchmark moe-grouping --model /path/to/model.gguf --nodes 2
mesh-llm benchmark moe-micro-analyze --model /path/to/model.gguf --prompts evals/moe/prompts/mt-bench-8.jsonl
```

## Live Runtime Examples

These new flags are meant for live MoE split experiments:

Full analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking analyze \
  --moe-grouping shared-core
```

Micro analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking micro-analyze \
  --moe-micro-prompt-count 1 \
  --moe-micro-tokens 8 \
  --moe-micro-layers all \
  --moe-grouping shared-core
```

Heuristic max + snake draft:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking heuristic-max \
  --moe-grouping snake-draft \
  --moe-replicate 256
```

Sequential fallback + shared core overlap:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking sequential \
  --moe-grouping shared-core \
  --moe-overlap 1
```
