set -euo pipefail
export PYTHONUNBUFFERED=1

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "☁️ Starting mesh-llm MoE HF job"
echo "📦 Model: $MODEL_REF"
echo "🗂️ Dataset repo: $DATASET_REPO"
echo "📥 Release URL: $MESH_LLM_RELEASE_URL"
echo "📁 Source repo: ${SOURCE_REPO:-unknown}"
echo "🧷 Source revision: ${SOURCE_REVISION:-unknown}"
echo "📄 Source file: ${SOURCE_FILE:-unknown}"
if [ -n "${HF_JOB_FLAVOR_PRETTY:-}" ] && [ -n "${HF_JOB_MAX_COST_USD:-}" ]; then
  echo "🖥️ Hardware: ${HF_JOB_FLAVOR_PRETTY} (${HF_JOB_FLAVOR:-unknown})"
  echo "💵 Pricing: \$${HF_JOB_UNIT_COST_USD:-unknown}/${HF_JOB_UNIT_LABEL:-unit}"
  echo "⏱️ Timeout: ${HF_JOB_TIMEOUT_SECONDS:-unknown}s"
  echo "🧮 Max cost at timeout: \$${HF_JOB_MAX_COST_USD}"
fi

cd "$workdir"
python3 - <<'PY'
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

url = os.environ["MESH_LLM_RELEASE_URL"]
archive = Path("mesh-llm-release.tar.gz")
bundle = Path("bundle")
with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
    shutil.copyfileobj(response, handle)
bundle.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(bundle, filter="data")
entries = [entry for entry in bundle.iterdir() if entry.exists()]
bundle_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else bundle
Path(".bundle-root").write_text(str(bundle_root))
PY

cd "$(cat "$workdir/.bundle-root")"
chmod +x ./mesh-llm ./llama-moe-analyze
export PATH="$PWD:$PATH"
cuda_lib_paths=(
  "$PWD"
  "/usr/local/cuda/lib64"
  "/usr/local/cuda/targets/x86_64-linux/lib"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
  "/usr/lib/x86_64-linux-gnu"
  "/usr/lib/wsl/lib"
)
ld_parts=()
for path in "${cuda_lib_paths[@]}"; do
  if [ -d "$path" ]; then
    ld_parts+=("$path")
  fi
done
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  ld_parts+=("$LD_LIBRARY_PATH")
fi
export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${ld_parts[*]}")"

echo "🔍 Verifying bundled binaries"
./mesh-llm --version
./llama-moe-analyze --help >/dev/null
echo "✅ Bundle verification complete"

echo "📥 Installing Hugging Face CLI"
python3 -m pip install -q --no-cache-dir -U huggingface_hub hf_xet

echo "📥 Downloading exact GGUF with hf"
model_dir="$workdir/model"
mkdir -p "$model_dir"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=64
export HF_HUB_DISABLE_TELEMETRY=1
MODEL_LOCAL_PATH="$(
  hf download "$SOURCE_REPO" "$SOURCE_FILE" \
    --repo-type model \
    --revision "$SOURCE_REVISION" \
    --local-dir "$model_dir"
)"
export MODEL_LOCAL_PATH
echo "✅ Model downloaded to $MODEL_LOCAL_PATH"

echo "🧠 Running analyze step"
stdbuf -oL -eL bash -lc '__ANALYZE_COMMAND__'
echo "✅ Analyze step complete"

echo "📤 Opening dataset PR"
stdbuf -oL -eL bash -lc '__SHARE_COMMAND__'
echo "✅ Dataset PR step complete"
