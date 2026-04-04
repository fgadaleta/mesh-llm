#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${VALIDATION_RESULTS_ROOT:-$REPO_ROOT/MLX_VALIDATION_RESULTS}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CASES_FILTER=""
SKIP_BUILD=0
MATRIX_FILE="$REPO_ROOT/scripts/mlx-parity-exact.tsv"
PROMPT="Reply with exactly: blue"
PROMPT_SUITE='[{"label":"alt-green","prompt":"Reply with exactly: green","expect_exact":"green"},{"label":"alt-red","prompt":"Reply with exactly: red","expect_exact":"red"}]'

usage() {
    cat <<'EOF'
Usage: scripts/run-mlx-parity-matrix.sh [options]

Run the exact GGUF/MLX backend parity matrix locally on macOS, preserving raw
artifacts for every case.

Options:
  --stamp <name>       Results stamp directory name (default: timestamp)
  --root <path>        Results root directory (default: ./MLX_VALIDATION_RESULTS)
  --cases <csv>        Comma-separated case ids to run
  --skip-build         Skip the initial `just build`
  -h, --help           Show this help

Outputs:
  <root>/<stamp>/exact-summary.tsv
  <root>/<stamp>/<case-id>/{stdout.log,stderr.log,mesh.log,...}
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --stamp)
            STAMP="$2"
            shift 2
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --cases)
            CASES_FILTER="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ]; then
    echo "❌ This suite currently requires macOS because it exercises MLX." >&2
    exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
    echo "❌ Missing matrix definition: $MATRIX_FILE" >&2
    exit 1
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    just build
fi

SUMMARY_DIR="$ROOT/$STAMP"
SUMMARY_FILE="$SUMMARY_DIR/exact-summary.tsv"
mkdir -p "$SUMMARY_DIR"
printf 'case\tbackend\texit\n' > "$SUMMARY_FILE"

append_summary() {
    local case_id="$1"
    local backend="$2"
    local rc="$3"
    python3 - "$SUMMARY_FILE" "$case_id" "$backend" "$rc" <<'PY'
import sys
path, case_id, backend, rc = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
header, rows = lines[0], lines[1:]
rows = [row for row in rows if not row.startswith(case_id + "\t")]
rows.append(f"{case_id}\t{backend}\t{rc}")
with open(path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for row in rows:
        f.write(row + "\n")
PY
}

want_case() {
    local case_id="$1"
    if [ -z "$CASES_FILTER" ]; then
        return 0
    fi
    python3 - "$CASES_FILTER" "$case_id" <<'PY'
import sys
wanted = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
case_id = sys.argv[2]
raise SystemExit(0 if case_id in wanted else 1)
PY
}

download_gguf_path() {
    local model_ref="$1"
    local output
    output=$(./target/release/mesh-llm models download "$model_ref" --gguf)
    printf '%s\n' "$output" >&2
    printf '%s\n' "$output" | awk '
        {
            line = $0
            sub(/^[[:space:]]+/, "", line)
        }
        line ~ /^\// && line ~ /\.gguf$/ { path = line }
        END {
            if (path != "") {
                print path
            } else {
                exit 1
            }
        }
    '
}

run_case() {
    local backend="$1"
    local case_id="$2"
    local model_ref="$3"
    local template_source="$4"
    local rc

    just stop >/dev/null 2>&1 || true

    if [ "$backend" = "gguf" ]; then
        local gguf_path
        gguf_path="$(download_gguf_path "$model_ref")"
        set +e
        VALIDATION_RESULTS_ROOT="$ROOT" \
        VALIDATION_RESULTS_STAMP="$STAMP" \
        scripts/run-validation-case.sh \
            "$backend" \
            "$case_id" \
            scripts/ci-gguf-smoke-test.sh \
            target/release/mesh-llm \
            llama.cpp/build/bin \
            "$gguf_path" \
            "$PROMPT" \
            "blue" \
            "" \
            "blue" \
            "$PROMPT_SUITE"
        rc=$?
        set -e
    else
        set +e
        VALIDATION_RESULTS_ROOT="$ROOT" \
        VALIDATION_RESULTS_STAMP="$STAMP" \
        scripts/run-validation-case.sh \
            "$backend" \
            "$case_id" \
            scripts/ci-mlx-smoke-test.sh \
            target/release/mesh-llm \
            "$model_ref" \
            "$template_source" \
            "$PROMPT" \
            "blue" \
            "" \
            "" \
            "blue" \
            "$PROMPT_SUITE"
        rc=$?
        set -e
    fi

    append_summary "$case_id" "$backend" "$rc"
}

while IFS=$'\t' read -r backend case_id model_ref template_source; do
    if [ "$backend" = "backend" ]; then
        continue
    fi
    if ! want_case "$case_id"; then
        continue
    fi
    echo ""
    echo "=== Running $case_id ($backend) ==="
    run_case "$backend" "$case_id" "$model_ref" "$template_source"
done < "$MATRIX_FILE"

echo ""
echo "=== Exact parity summary ==="
cat "$SUMMARY_FILE"
echo ""
echo "Raw artifacts: $SUMMARY_DIR"
