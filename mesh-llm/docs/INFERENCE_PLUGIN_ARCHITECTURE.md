# Inference Plugin Architecture

This note defines the architectural split for moving the current `llama` and MLX integration toward plugins without moving mesh orchestration policy into plugins.

## Goal

Move backend-specific inference runtime behavior out of core while keeping mesh-wide planning, routing, and failover decisions in `mesh-llm`.

The intended result is:

- `mesh-llm` owns planning and routing
- plugins own backend execution
- backend-specific analysis tools can be exposed through plugins
- MLX and llama fit the same high-level contract even though they have different runtime shapes

## What Stays In Core

Core should continue to own:

- mesh election and assignment
- target routing for `/v1/*`
- endpoint inventory and health
- served-model advertisement and gossip
- MoE placement planning
- MoE fallback and failover planning
- ranking policy decisions
- capability/provider selection

This means the logic in the current and in-progress MoE work should stay host-owned:

- choosing when to use cached, imported, micro-analyze, or full ranking
- choosing active shard sets
- choosing fallback full-coverage sets
- deciding when a new plan materially improves the running plan
- exposing target sets to the proxy

These are product and mesh policy decisions, not backend implementation details.

## What Moves Into Plugins

Plugins should own backend-specific execution:

- model loading
- backend startup and shutdown
- local serving endpoint creation
- backend-specific helper processes
- backend-specific launch flags and heuristics
- backend-specific analysis tooling

For the current llama path, that includes:

- `llama-server`
- `rpc-server`
- `llama-moe-analyze`
- tensor split argument shaping
- draft model wiring
- mmproj wiring
- worker-side helper lifecycle

For MLX, that includes:

- in-process model load
- in-process OpenAI-compatible HTTP serving
- MLX-specific runtime and template behavior

## Core Contracts

The split should use two plugin-facing contracts.

### 1. `InferenceEndpointProvider`

This is the runtime-serving contract.

It should support:

- `start_local_runtime`
- `start_distributed_host_runtime`
- `start_distributed_worker_runtime`
- `stop_runtime`
- `describe_runtime`

The output should be host-usable runtime metadata:

- endpoint descriptor
- backend label
- context length
- instance id
- health/readiness state

The endpoint descriptor should fit the existing plugin endpoint registration model, so `mesh-llm` continues to route through the same host-owned inference endpoint registry.

### 2. `MoeRankingProvider`

This is optional and backend-specific.

It should support:

- inspect model topology when backend-specific tooling is required
- produce ranking artifacts
- report provenance and strategy metadata
- import or export ranking artifacts if needed

This contract is especially relevant for llama/GGUF because the ranking tools are backend-specific. MLX probably does not implement this initially.

## Core Planner

Core should own a planner layer that consumes runtime and ranking providers but does not depend on any specific backend implementation.

Conceptually:

- `MoePlanner`
- `PlacementPlan`
- `FailoverPlan`
- `RankingPolicy`

`MoePlanner` decides:

- whether a model is MoE
- which ranking strategy to use
- which nodes are active shards
- which nodes are full fallbacks
- when to replace a running plan

Plugins provide capabilities and artifacts. Core decides how to use them.

## Llama Split

Llama should be split in two phases.

### Phase 1: local runtime plugin

Move the local runtime path behind `InferenceEndpointProvider`:

- local `llama-server` startup
- local process lifecycle
- endpoint registration

This replaces the hard-coded local launch path in `runtime/local.rs`.

### Phase 2: distributed llama runtime plugin

Move distributed host and worker runtime behavior into the same llama plugin:

- host `llama-server` startup
- worker `rpc-server` startup
- backend-specific split launch arguments
- backend-specific analysis tool invocation

Core election still decides host vs worker vs shard placement. The plugin only executes the assigned role.

## MLX Split

MLX already resembles the right boundary more closely.

The MLX PR shape is effectively:

- same local OpenAI-compatible surface
- different runtime implementation
- different startup path
- no change to proxy semantics

That makes MLX a straightforward `InferenceEndpointProvider` plugin:

- local only, initially
- plugin-hosted endpoint
- no distributed worker/runtime contract at first
- no `MoeRankingProvider` at first

## Why The MoE Branch Matters

The MoE benchmark and placement branch strengthens this split.

That branch adds richer host policy for:

- ranking provenance
- peer-imported ranking
- micro-analyze fallback
- fallback coverage sets
- placement improvement rules
- startup quiet periods and failover behavior

Those are planner concerns and should remain in core.

What should move out of core from the MoE side is backend-specific ranking production, not ranking policy.

So:

- ranking policy stays in core
- ranking generation can move into a backend plugin

## Recommended Migration Order

1. Define the plugin contracts:
   - `InferenceEndpointProvider`
   - `MoeRankingProvider`

2. Extract a small host-owned planner layer from the current inference/MoE logic:
   - placement
   - failover
   - ranking policy

3. Pluginize the current local llama runtime behind `InferenceEndpointProvider`.

4. Port MLX onto the same `InferenceEndpointProvider` contract.

5. Move distributed llama host/worker runtime behavior into the llama plugin.

6. Move llama/GGUF-specific ranking production behind `MoeRankingProvider`.

7. Keep proxy routing and `/v1/*` host-owned throughout.

## Practical Rule

If a piece of logic answers:

- "where should requests go?"
- "which nodes should serve?"
- "when should we fail over?"
- "which ranking should we trust?"

it belongs in core.

If a piece of logic answers:

- "how do I start this backend?"
- "how do I serve this model?"
- "how do I generate backend-specific ranking artifacts?"

it belongs in a plugin.
