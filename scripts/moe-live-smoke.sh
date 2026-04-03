#!/usr/bin/env bash
# moe-live-smoke.sh — validate an already-running MoE deployment end-to-end.
#
# Usage:
#   scripts/moe-live-smoke.sh [--expected-nodes N] [--timeout SEC] <model> <api-url> <console-url> [console-url...]
#
# The script waits for each console endpoint to report the model as warm with the
# expected node_count, verifies the model is listed on the chosen API endpoint,
# and finally runs one /v1/chat/completions request through that API.

set -euo pipefail

EXPECTED_NODES=2
TIMEOUT=120

usage() {
    cat <<'EOF'
Usage: scripts/moe-live-smoke.sh [--expected-nodes N] [--timeout SEC] <model> <api-url> <console-url> [console-url...]

Arguments:
  model         Model id as exposed via /v1/models
  api-url       Base API URL, e.g. http://studio54.local:9337
  console-url   Base console URL or direct /api/status URL, e.g. http://studio54.local:3131

Options:
  --expected-nodes N  Require this node_count and at least this many active_nodes (default: 2)
  --timeout SEC       Per-console wait timeout in seconds (default: 120)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expected-nodes)
            EXPECTED_NODES="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 1
fi

MODEL="$1"
API_URL="${2%/}"
shift 2
STATUS_URLS=("$@")

normalize_status_url() {
    local url="${1%/}"
    if [[ "$url" == */api/status ]]; then
        printf '%s\n' "$url"
    else
        printf '%s/api/status\n' "$url"
    fi
}

wait_for_status() {
    local raw_url="$1"
    local status_url
    status_url="$(normalize_status_url "$raw_url")"
    local last_summary="status unavailable"

    echo "Checking ${status_url} for ${MODEL}..."
    for ((i = 1; i <= TIMEOUT; i++)); do
        local payload
        payload="$(curl -sf --max-time 5 "$status_url" 2>/dev/null || true)"
        if [[ -n "$payload" ]]; then
            local summary
            summary="$(printf '%s' "$payload" | python3 -c '
import json
import sys

model = sys.argv[1]
expected = int(sys.argv[2])

try:
    status = json.load(sys.stdin)
except Exception:
    print("0|parse-error")
    raise SystemExit(0)

for entry in status.get("mesh_models", []):
    if entry.get("name") == model:
        active = entry.get("active_nodes") or []
        ready = (
            entry.get("status") == "warm"
            and int(entry.get("node_count", 0)) >= expected
            and len(active) >= expected
        )
        print(
            "{}|status={} nodes={} active={}".format(
                int(ready),
                entry.get("status"),
                entry.get("node_count", 0),
                len(active),
            )
        )
        raise SystemExit(0)

print("0|missing")
' "$MODEL" "$EXPECTED_NODES")"
            local ready="${summary%%|*}"
            last_summary="${summary#*|}"
            if [[ "$ready" == "1" ]]; then
                echo "  ✅ ${last_summary}"
                return 0
            fi
        fi

        if (( i == TIMEOUT )); then
            echo "  ❌ ${last_summary}" >&2
            return 1
        fi

        if (( i == 1 || i % 10 == 0 )); then
            echo "  waiting... ${last_summary}"
        fi
        sleep 1
    done
}

for status_url in "${STATUS_URLS[@]}"; do
    wait_for_status "$status_url"
done

echo "Checking ${API_URL}/v1/models..."
MODELS_PAYLOAD="$(curl -sf --max-time 10 "${API_URL}/v1/models")"
printf '%s' "$MODELS_PAYLOAD" | python3 -c '
import json
import sys

model = sys.argv[1]
payload = json.load(sys.stdin)
names = [entry.get("id") for entry in payload.get("data", [])]
if model not in names:
    raise SystemExit(f"model {model!r} not found in /v1/models: {names}")
print(f"  ✅ /v1/models includes {model}")
' "$MODEL"

echo "Testing ${API_URL}/v1/chat/completions..."
CHAT_PAYLOAD="$(curl -sf --max-time 90 "${API_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: moe smoke ok\"}],
        \"max_tokens\": 12,
        \"temperature\": 0
    }")"
printf '%s' "$CHAT_PAYLOAD" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
content = payload["choices"][0]["message"].get("content", "").strip()
if not content:
    raise SystemExit("empty chat response")
print(f"  ✅ chat response: {content}")
'

echo "MoE live smoke passed for ${MODEL}"
