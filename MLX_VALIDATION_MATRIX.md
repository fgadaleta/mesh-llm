# MLX Validation Matrix

Local-first backend-parity ledger for model families. The point is not to judge
MLX in isolation; it is to compare `🦙 GGUF` against `🍎 MLX` on the same family /
model / case so we can tell shared model weakness from MLX-specific regressions.

## Legend

| Status | Meaning |
|---|---|
| `PASS` | Validated locally and behaved acceptably for the checks listed |
| `FAIL` | Reproduced a real issue locally |
| `PARTIAL` | Loads and answers basic prompts, but has behavior issues or incomplete coverage |
| `BLOCKED` | Could not be validated locally on this machine |
| `PENDING` | Not checked yet |

## GGUF Parity

| Status | Meaning |
|---|---|
| `MATCH` | GGUF showed the same behavior, so the issue is likely not MLX-specific |
| `MLX WORSE` | GGUF handled the same case better than MLX |
| `MLX BETTER` | MLX handled the same case better than GGUF |
| `PENDING` | GGUF comparison not run yet |
| `BLOCKED` | Could not get a meaningful GGUF comparison locally |

## Pair Quality

| Status | Meaning |
|---|---|
| `HIGH` | Same family, same size, same instruct/chat target, and close quant class; good parity signal |
| `MEDIUM` | Same family and roughly same target, but quant or conversion path differs materially |
| `LOW` | Only approximate family parity; useful for triage, but not a strong apples-to-apples comparison |
| `PENDING` | Pair quality not assessed yet |

## Models

| Family | Model Pair | GGUF Target | MLX Target | Pair Quality | Last Checked | GGUF Exact | MLX Exact | GGUF Behavior | MLX Behavior | Parity | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 0.5B instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | `HIGH` | 2026-04-04 | `PASS` | `FAIL` | `PARTIAL` | `FAIL` | `MLX WORSE` | `PARTIAL` | Rebuilt-engine exact rerun confirmed the earlier result. GGUF again passed `blue / green / red`; MLX again returned `红` for exact `red`. Full MLX behavior run failed 37/80 prompts. Sampled GGUF behavior was materially stronger on the same early writing and roleplay blocks, so this is not just shared model weakness. |
| Qwen3 | 0.6B instruct | `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | `Qwen/Qwen3-0.6B-MLX-8bit` | `HIGH` | 2026-04-05 | `FAIL` | `FAIL` | `PENDING` | `FAIL` | `MATCH` | `PARTIAL` | Reran the checked-in matrix on `studio54.local` with a closer `Q8_0` vs `8bit` pair. Both backends passed `blue / green / red` and then failed the same `primary-colors` prompt by emitting `rgb` shorthand instead of the full words `red, green, blue`: GGUF returned `rgb,rgb,rgb`, MLX also returned `rgb,rgb,rgb`. On this better-matched pair, the exact failure now reads as shared model weakness rather than an MLX-specific regression. Sampled MLX behavior is still weak on broader writing, roleplay, and reasoning prompts. |
| Llama | 3.2 1B instruct | `llmware/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `HIGH` | 2026-04-04 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Same family, same size, same instruct target, close quant class. Exact parity says this is shared model behavior, not MLX-specific: both backends loaded cleanly but returned `Blue` instead of exact lowercase `blue` on the first deterministic prompt. |
| Gemma 2 | 2B instruct | `bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf` | `mlx-community/gemma-2-2b-it-4bit` | `HIGH` | 2026-04-04 | `PASS` | `FAIL` | `PENDING` | `PENDING` | `MLX WORSE` | `FAIL` | Rebuilt-engine exact rerun confirmed the earlier result. GGUF exact suite passed `blue / green / red`. A targeted raw-log rerun preserved the MLX startup failure: `MLX server failed: parsing config.json`, followed by `Error: early eof`. This remains a clear MLX-specific regression rather than shared model weakness. |
| Gemma 3 | 1B instruct | `meshllm/gemma-3-1b-it-parity-f16-gguf/gemma-3-1b-it-f16.gguf` | `meshllm/gemma-3-1b-it-parity-bf16-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Published same-origin high-fidelity parity pair derived from `google/gemma-3-1b-it`. Validated on `studio54.local`: both backends passed the full exact suite with identical outputs, including `primary-colors -> Red, Green, Blue`. This replaces the noisier public low-bit Gemma3 pair for future parity checks. |
| Gemma 4 | E4B instruct | `unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` | `HIGH` | 2026-04-04 | `FAIL` | `PASS` | `PENDING` | `PENDING` | `MLX BETTER` | `PARTIAL` | Rebuilt-engine rerun still could not get GGUF to a ready state. A targeted raw-log rerun preserved the repeated startup pattern: `Running as host`, `Starting llama-server...`, `Serving entirely (19GB VRAM)`, then `Error: early eof`, with no ready signal. MLX exact suite passed `blue / green / red` using `chat_template.jinja`, so MLX is operationally better on this pair right now. |
| GLM4 | 9B 0414 | `lmstudio-community/GLM-4-9B-0414-GGUF/GLM-4-9B-0414-Q4_K_M.gguf` | `mlx-community/GLM-4-9B-0414-4bit` | `HIGH` | 2026-04-04 | `PASS` | `PASS` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Rebuilt-engine exact rerun confirmed parity. Both backends passed `blue / green / red`, though both emit a leading newline before the color. Exact parity is still good here. |
| LFM2 | 350M | `LiquidAI/LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf` | `mlx-community/LFM2-350M-4bit` | `HIGH` | 2026-04-04 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Rebuilt-engine exact rerun confirmed the earlier result. GGUF again failed the first exact prompt by answering with explanatory prose instead of `blue`, while MLX again passed `blue / green` but answered exact `red` as `Red.`. This still looks like shared tiny-model weakness rather than an MLX-specific regression. |
| DeepSeekV3 / Kimi-K2 | K2 instruct | `public GGUF target TBD` | `mlx-community/Kimi-K2-Instruct-4bit` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Public GGUF target is still unresolved and likely impractical locally, so parity here will be approximate even when we can run it. |
| gpt-oss | 20B-ish | `unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q2_K.gguf` | `concrete MLX target TBD` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Need concrete MLX repo target before this becomes a meaningful parity pair. |
| Kimi Linear | 48B A3B | `public GGUF target TBD` | `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Public GGUF target is unresolved and likely too large locally, so parity will be approximate even if we can run it. |

## Notes

- Exact smoke means the deterministic `blue / green / red` style suite plus reasoning-on probe where relevant.
- Behavior means the MT-Bench-derived behavior harness in [`scripts/ci-mt-bench-behavior.py`](/Users/jdumay/.codex/worktrees/e497/mesh-llm/scripts/ci-mt-bench-behavior.py).
- Raw rebuilt-engine exact rerun artifacts are stored under [`MLX_VALIDATION_RESULTS/rerun-20260404-buildsync`](/Users/jdumay/.codex/worktrees/e497/mesh-llm/MLX_VALIDATION_RESULTS/rerun-20260404-buildsync).
- The judgment rule is simple:
  - `🦙 GGUF FAIL` + `🍎 MLX FAIL` = probably shared model weakness
  - `🦙 GGUF PASS` + `🍎 MLX FAIL` = MLX-specific problem and not OK
  - `🦙 GGUF FAIL` + `🍎 MLX PASS` = MLX at least not worse there
- Record enough detail in `Notes` to make the next fix obvious.
