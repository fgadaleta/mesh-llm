# mesh-llm TODO

## SSD Expert Streaming

Run giant MoE models on a single node by streaming active experts from NVMe instead of fitting everything in RAM. Trunk (attention, norms, embeddings) stays resident; expert weights live on disk, `pread()`'d on demand per token.

[flash-moe](https://github.com/danveloper/flash-moe) already does this. From-scratch C/Metal engine, runs Qwen3.5-397B-A17B (120GB 2-bit experts) at 5.5 tok/s on a 48GB M3 Max. 6GB resident memory. Complete working inference engine — 7K lines of C/ObjC/Metal. See also [ROADMAP.md](../ROADMAP.md).

**Plan:** Use flash-moe directly as an alternative backend. Mesh-llm spawns it like it spawns llama-server — process management, HTTP/SSE wrapper, kill on shutdown. Don't try to hack SSD streaming into llama.cpp; the `ggml_mul_mat_id` op assumes all expert weights are resident in one contiguous tensor per layer. Changing that is deep surgery across ggml, the Metal backend, and the model loader. Not worth it when a working engine exists.

**Limitation:** flash-moe only supports Qwen3.5-397B (GatedDeltaNet + full attention, 512 experts, hardcoded architecture). That's the model we want to run. More models = more forward pass implementations.

**What flash-moe needs to integrate:**
- HTTP/SSE endpoint (currently interactive CLI only — `chat.m`)
- OpenAI-compatible `/v1/chat/completions` so mesh-llm's proxy can route to it
- Model weight prep: `repack_experts.py` to convert safetensors → packed per-layer binary files

**Key findings from flash-moe (don't repeat their mistakes):**
- Trust the OS page cache — every custom cache made it worse. Deleting the cache was a 38% speedup.
- `pread()` >> `mmap()` for expert loading (5×). mmap = 240 page faults per 3.9MB expert.
- 2-bit expert quant preserves quality (RMSE ~0.001). 44% smaller, biggest throughput win.
- Kernel I/O hints (F_RDADVISE, MADV_RANDOM, etc.) useless or harmful on Apple Silicon.
- Speculative routing doesn't work — 65-80% wrong predictions, wastes bandwidth.

## MoE Expert Sharding

Design: [MoE_PLAN.md](../MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](../MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](../MoE_SPLIT_REPORT.md)

- [x] Phase 1–3: Routing analysis, expert masking, mesh integration. Tested OLMoE-1B-7B over WAN.
- [ ] **Phase 4: lazy `moe-analyze`** — auto-run ranking for unknown MoE models.
- [ ] **Phase 6: scale testing** — Mixtral 8×22B, Qwen3-235B-A22B.

## Smart Router
- [ ] **Static speed estimates**: Add `tok_s: f64` to ModelProfile. Feed into scoring so Quick tasks prefer fast models.
- [ ] **Response quality checks**: Detect empty/repetitive/truncated responses, trigger retry with different model.

## Resilience
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.
- [ ] **`kill_llama_server()` uses `pkill -f`**: Should kill by PID, not pattern match.

## Experiments
- [ ] Qwen3.5-397B-A17B on single 128GB M4 Max (SSD streaming)
- [ ] Qwen3.5-397B-A17B across 128GB M4 Max + second machine (MoE split)
- [ ] Largest dense models across 2+ machines (Llama-3.3-70B, Qwen2.5-72B)
