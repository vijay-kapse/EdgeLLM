#!/usr/bin/env bash
# Reproduce the full EdgeLLM pipeline end-to-end on this machine.
#
# Prereqs:
#   - Python 3.10-3.12 venv with:  pip install -e ".[onnx,eval,report]"
#   - For the C++ harness (optional): cmake + onnxruntime (macOS: brew install cmake onnxruntime)
#   - For the Snapdragon NPU (optional): pip install -e ".[aihub]" and an AI Hub token
#
# Usage:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT="Explain what neural network quantization is, in one short paragraph."

echo "==> Environment"
edgellm info

echo "==> Baseline generation (PyTorch)"
edgellm generate -p "$PROMPT" --greedy --max-new-tokens 48

echo "==> ONNX export (FP32)"
edgellm export

echo "==> Quantize to INT8 + INT4"
edgellm quantize

echo "==> Benchmark all precisions (FP32 / INT8 / INT4)"
edgellm benchmark

echo "==> Regenerate table + charts"
edgellm report

# Resolve the exported INT8 model directory from the configured model id.
MODEL_DIR=$(python - <<'PY'
from pathlib import Path
from edgellm.config import Config
cfg = Config.from_yaml("configs/default.yaml")
safe = cfg.model.id.replace("/", "__")
print(Path(cfg.export.output_dir) / f"{safe}-int8-dynamic")
PY
)

echo "==> C++ inference harness (optional; requires cmake + onnxruntime)"
if command -v cmake >/dev/null 2>&1; then
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build
  IDS=$(edgellm encode -p "$PROMPT")
  ./cpp/build/edgellm_infer --model "$MODEL_DIR" --tokens "$IDS" --max-new-tokens 48
else
  echo "   cmake not found; skipping C++ harness."
fi

echo "==> Qualcomm Snapdragon NPU (optional; requires an AI Hub token)"
python aihub/run_on_snapdragon.py --model "$MODEL_DIR" || true

echo "==> Done. See results/benchmark.md and results/benchmark_chart.png"
