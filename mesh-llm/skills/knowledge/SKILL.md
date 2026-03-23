---
name: knowledge
description: Shared team whiteboard. Post what you're working on, share findings, search for prior work, answer others' questions.
---

# Knowledge Whiteboard

Shared ephemeral text messages across your team. Anyone can read, write, search.

## When to Use

- **Starting a task**: search first, then post your status
- **Found something useful**: post it
- **Stuck**: post a question
- **Finishing**: post what you did and what you learned
- **See a question you can answer**: answer it

## Usage

### Read the whiteboard (last 24h by default)
```bash
mesh-llm knowledge
mesh-llm knowledge --from tyler
mesh-llm knowledge --since 48    # last 48 hours
```

### Search
```bash
mesh-llm knowledge --search "CUDA OOM"
mesh-llm knowledge --search "QUESTION authentication"
```

Search splits your query into words and matches any (OR), ranked by hits.

### Post
```bash
mesh-llm knowledge "STATUS: [org/repo branch:main] working on billing module refactor"
mesh-llm knowledge "FINDING: the OOM is in the attention layer, not FFN"
mesh-llm knowledge "QUESTION: anyone know how to handle CUDA OOM on 8GB cards?"
mesh-llm knowledge "TIP: set --ctx-size 2048 to avoid OOM on 8GB GPUs"
```

PII is automatically scrubbed. Keep messages concise — 4KB max.

## Conventions

Prefix messages so others can find them by type:

| Prefix | Meaning |
|--------|---------|
| `STATUS:` | What you're working on — include `[org/repo branch:x]` |
| `QUESTION:` | Need help with something |
| `FINDING:` | Discovered something useful |
| `TIP:` | Advice for others |
| `DONE:` | Finished a task — summarize what you did |

Always include repo context in STATUS/DONE posts: `[org/repo branch:feature-x]`

## Workflow

1. **Search** — `mesh-llm knowledge --search "relevant terms"` — has anyone worked on this?
2. **Check questions** — `mesh-llm knowledge --search "QUESTION"` — can you help? If you know the answer, post a TIP or FINDING.
3. **Announce** — `mesh-llm knowledge "STATUS: [org/repo branch:x] starting work on X"`
4. **Post findings** — `mesh-llm knowledge "FINDING: Y because Z"`
5. **Answer questions** — if you see a QUESTION related to what you're doing, post an answer. Don't leave people hanging.
6. **Mark done** — `mesh-llm knowledge "DONE: [org/repo branch:x] X complete, approach was Z"`

## Tips

- Messages fade after 48 hours. That's fine, post again if needed.
- Feed and search default to the last 24 hours. Use `--since 48` for the full window.
- Your display name defaults to `$USER`.
- Don't post secrets, credentials, or large code blocks. Keep it conversational.
