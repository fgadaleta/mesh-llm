# Plugins

This document defines the `mesh-llm` plugin architecture.

It describes the target architecture, not just the code as it exists today.

As implementation lands, this document should be updated to match the intended end state and the concrete protocol and runtime decisions that have been made.

The main goals are:

- keep `mesh-llm` decoupled from specific plugins
- let bundled plugins be auto-registered without special-casing product behavior
- make MCP and HTTP first-class host projections
- support large request and response bodies without blocking control traffic
- keep plugin author boilerplate low

## Design Summary

A plugin is a local service process launched by `mesh-llm`.

The system has three core pieces:

- one long-lived control connection per plugin process
- zero or more short-lived negotiated streams for large or streaming data
- one declarative plugin manifest that the host `stapler` projects into MCP, HTTP, and optional promoted product APIs

`mesh-llm` remains the owner of:

- plugin lifecycle
- local IPC
- stapling manifest-declared services onto host-facing protocols
- HTTP serving
- MCP serving
- capability routing
- mesh participation and peer-to-peer transport

A plugin owns:

- its own feature logic
- local state
- operation handlers
- resource handlers
- prompt handlers
- plugin-specific mesh channel semantics

Plugins do not need to implement raw MCP or raw HTTP servers.

The `stapler` is the host projection layer that turns plugin manifests into exposed MCP and HTTP surfaces.

## High-Level Model

The plugin system is projection-oriented at the DSL level and service-oriented at the runtime level.

Plugin authors think in terms of the host surfaces they contribute to:

- `mcp`
- `http`
- `inference`
- `provides`

The host runtime still executes native service invocations internally, but the author-facing DSL is organized by the surface the plugin contributes to.

This means:

- local MCP tools, resources, prompts, and completions live under `mcp`
- attached external MCP servers also live under `mcp`
- local HTTP routes live under `http`
- attached or plugin-hosted inference backends live under `inference`
- stable product capabilities live under `provides`

There is no separate top-level `services` section in the preferred DSL.

## Core Principles

### 1. Bundled Plugins Are Allowed

Plugins shipped in this source tree may be auto-registered by the host.

That is acceptable coupling.

What is not acceptable is embedding one plugin's runtime behavior directly into core mesh logic. Core mesh transport and state should stay generic.

### 2. One Control Connection, Many Data Streams

Each plugin process has one long-lived control connection.

Use the control connection for:

- initialize / health / shutdown
- manifest registration
- small RPC-style requests
- mesh event delivery
- stream negotiation
- cancellation

Do not use the control connection for large uploads, downloads, or long-lived streaming responses.

For large or streaming payloads, the host and plugin negotiate a short-lived side stream.

### 3. MCP Is A Host Projection

`mesh-llm` is the MCP server.

Plugins do not need to implement MCP JSON-RPC directly. They declare MCP-facing services in the manifest, and the host `stapler` exposes them over MCP.

### 4. HTTP Is A Host Projection

`mesh-llm` owns the HTTP server.

Plugins may declare HTTP bindings, but they do not need to run an HTTP server themselves. The host `stapler` maps HTTP requests onto plugin operations and resources.

### 5. Capabilities Are Stable Product Contracts

When `mesh-llm` wants a stable product API such as `/api/objects`, core should depend on a named capability like `object-store.v1`, not on a specific plugin ID like `blobstore`.

## Architecture

### Control Session

There is one long-lived control session between host and plugin.

The control session is used for:

- plugin startup and manifest exchange
- health checks
- native service invocation requests and responses
- plugin-to-host notifications
- host-to-plugin mesh events
- opening and closing streams
- cancellation and error reporting

The control session should stay responsive even while the plugin is sending or receiving large payloads.

The native runtime contract is service-oriented, not MCP-oriented.

The host invokes services such as:

- operations
- prompts
- resources
- completions

MCP method names like `tools/call` and `prompts/get` are projection-layer concerns. They are not the preferred host/plugin runtime contract.

### Streams

Streams are short-lived negotiated channels for a single request, response, or transfer.

They are opened via the control session and then carry data independently.

Streams are used for:

- large HTTP request bodies
- large HTTP responses
- streaming uploads and downloads
- server-sent events or similar long-lived responses
- future bulk data flows between host and plugin

On Unix, streams map to short-lived Unix sockets.

On Windows, streams map to short-lived named pipes.

The protocol concept is `stream`, not `socket`, so the transport binding remains platform-specific.

### Why Streams Exist

The current single-socket framed-envelope design is vulnerable to head-of-line blocking. Even chunked transfer traffic still competes with health checks, tool calls, mesh events, and other control messages on the same queue.

This architecture avoids that by separating:

- control plane traffic
- bulk and streaming data traffic

## Manifest

On startup, a plugin returns a manifest that declares what it provides to the host.

Conceptually, the manifest contains:

- plugin identity and version
- provided capabilities
- MCP contributions
- HTTP contributions
- inference contributions
- any mesh channel declarations the plugin needs

The manifest is the source of truth for host projections.

## Plugin Author Experience

The primary design goal is very low boilerplate.

The preferred DSL is surface-first:

- `provides`
- `mcp`
- `http`
- `inference`

Each section is self-contained. If a plugin contributes something to a host surface, it is declared in the section for that surface.

Example:

```rust
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata,
    http::{get, post},
    inference::openai_http,
    mcp::{external_stdio, prompt, resource, tool},
};

let plugin = mesh_llm_plugin::plugin! {
    metadata: PluginMetadata::new(
        "notes",
        "1.0.0",
        plugin_server_info(
            "notes",
            "1.0.0",
            "Notes",
            "Shared notes services",
            None::<String>,
        ),
    ),

    provides: [
        capability("notes.v1"),
        capability("search.v1"),
    ],

    mcp: [
        tool("search")
            .description("Search notes")
            .input::<SearchArgs>()
            .handle(search),

        resource("notes://latest")
            .name("Latest Notes")
            .handle(read_latest),

        prompt("summarize_notes")
            .description("Summarize recent notes")
            .handle(summarize_notes),

        external_stdio("filesystem", "npx")
            .arg("-y")
            .arg("@modelcontextprotocol/server-filesystem"),
    ],

    http: [
        get("/search")
            .description("Search notes")
            .input::<SearchArgs>()
            .handle(search),

        post("/notes")
            .description("Create a note")
            .input::<PostArgs>()
            .handle(post_note),
    ],

    inference: [
        openai_http("local-llm", "http://127.0.0.1:8080/v1")
            .managed_by_plugin(false),
    ],
};
```

In this model:

- `mcp` contains both local MCP contributions and attached external MCP servers
- `http` contains local HTTP contributions
- `inference` contains both attached external inference endpoints and plugin-hosted inference providers
- `provides` declares stable capability contracts that core product routes can depend on

The runtime and `stapler` handle:

- schema exposure
- MCP projection
- HTTP projection
- request validation
- stream negotiation
- transport details
- host-side routing and aggregation

Plugin authors should not manually implement:

- MCP `tools/list`
- MCP `tools/call`
- MCP `resources/read`
- HTTP routing
- control-plane socket negotiation

### Streaming

Streaming is explicit in the DSL.

For HTTP bindings, the preferred modifiers are:

- `.stream_request()`
- `.stream_response()`
- `.sse()`

These declare whether the request body, response body, or response format requires side-stream transport.

## External Endpoints

Plugins may register external services without proxying all traffic through the plugin process.

This is a control-plane declaration, not a request proxying requirement.

In practice:

- attached external MCP servers are declared in the `mcp` section
- attached or plugin-hosted inference backends are declared in the `inference` section

`mesh-llm` then talks to those services directly when appropriate.

This keeps heavy data-plane traffic out of plugin IPC.

### MCP Contributions

The `mcp` section may contain both:

- local MCP-facing items implemented by the plugin
- attached external MCP servers

Preferred external forms include:

- `external_stdio(...)`
- `external_http(...)`
- `external_tcp(...)`
- `external_unix_socket(...)`

External MCP names are namespaced as:

- `plugin_name.method`

### Inference Contributions

The `inference` section may contain both:

- attached external OpenAI-compatible endpoints
- plugin-hosted inference providers

Preferred forms include:

- `openai_http(...)` for attached external endpoints
- `provider(...)` for plugin-hosted backends

### Why Endpoint Registration Exists

Some services already speak a protocol that `mesh-llm` knows how to use directly.

Examples:

- a local OpenAI-compatible inference server
- an external MCP server reachable over stdio, streamable HTTP, Unix socket, named pipe, or TCP
- a plugin-hosted inference runtime such as an MLX-backed local server

In these cases, the plugin should remain the control-plane owner for:

- discovery
- lifecycle
- readiness
- availability

But `mesh-llm` should own the data plane when possible.

This keeps large request/response traffic out of plugin IPC.

### Health And Availability

Endpoint health is separate from plugin health.

If an endpoint health check fails:

- the endpoint becomes unavailable
- the endpoint is removed from routing or aggregation
- the plugin remains loaded
- the plugin is not marked disabled
- the host keeps checking health

If health returns:

- the endpoint becomes available again automatically

This is important because a plugin may be healthy while its managed or discovered service is:

- starting
- restarting
- temporarily unhealthy
- reloading a model
- intentionally stopped

The host should treat plugin liveness and endpoint liveness as separate concerns.

### Recommended State Model

Conceptually, the system should track at least:

- plugin state
- endpoint state
- model or route availability

Suggested plugin states:

- `starting`
- `running`
- `degraded`
- `disconnected`
- `failed`

Suggested endpoint states:

- `unknown`
- `starting`
- `healthy`
- `unhealthy`
- `unavailable`

Suggested routed availability states:

- `advertised`
- `routable`
- `draining`
- `unavailable`

Routing decisions should depend on endpoint health, not just plugin process health.
