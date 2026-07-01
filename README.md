# EdgeLLM

**Quantize a small language model and run it on-device — CPU, GPU, mobile, and a Qualcomm Snapdragon NPU — with a C/C++ inference harness and a rigorous, honestly-measured benchmark suite.**

This is a portfolio project built around the requirements of a Qualcomm *Machine Learning Engineer (AI Research, GenAI for the Edge)* role. Every latency, throughput, memory, size, and accuracy number in this README comes from an **actual run** on real hardware. Steps that require hardware or credentials I have not yet wired up are shown as clearly-labeled `TODO(vijay): run on <device>` placeholders — never invented.

> Status: **Phase 2 complete** (ONNX export + FP32 baseline benchmark). Phases 3–8 in progress.

---

## What it does (target)

- Load a small instruct LLM (default **Qwen2.5-0.5B-Instruct**, swappable) and generate text.
- Export to **ONNX** and benchmark an FP32 baseline: latency, tokens/sec, peak RAM, on-disk size, perplexity.
- **Quantize** to INT8 (PyTorch + ONNX Runtime) and INT4 (GPTQ/AWQ, or GGUF fallback), and compare all precisions.
- Run inference from a **C++17** harness via the ONNX Runtime C++ API.
- Profile on a real **Qualcomm Snapdragon NPU** via Qualcomm AI Hub / the ONNX Runtime QNN Execution Provider.
- Optional stretch: a custom SIMD INT8 GEMM kernel and an Android on-device app.

## Requirement → feature mapping

| Qualcomm JD requirement | Where it shows up in EdgeLLM |
| --- | --- |
| Python, PyTorch, deep learning, GenAI | `edgellm/models.py`, `edgellm/runners.py` (Phase 1) |
| Model optimization / on-target deployment | ONNX export + benchmark harness (Phase 2) |
| Neural-network model optimization (quantization) | `edgellm/quantize.py` INT8/INT4 (Phase 3) |
| C/C++; analytical/debugging | `cpp/` ONNX Runtime C++ harness (Phase 4) |
| NPUs / ML accelerators | `aihub/`, QNN EP runner (Phase 5) |
| Usability, SW design, communication | CLI, config, CI, this README (Phase 0/6) |
| Optimization of algebraic ops for HW cores | `kernels/` SIMD INT8 GEMM (Phase 7, optional) |
| Android, on-device inference | `android/` ORT Mobile app (Phase 8, optional) |

## Benchmark results

Real numbers are filled in as each phase runs. Nothing here is invented.

**Model:** Qwen/Qwen2.5-0.5B-Instruct · **Machine:** Apple Silicon (arm64), macOS · **Workload:** 64 new tokens, greedy, warmup 2 / measured 5 · **Perplexity:** WikiText-2 (32-line slice, 512-token windows).

| Backend | Precision | Device | Size (MB) | Latency (s) | Throughput (tok/s) | Peak RAM (MB) | Perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch | fp32 | mps | 942.32 | 2.99 ± 0.05 | 21.37 | 2494.1 | 19.0 |
| ort-cpu | fp32 | cpu | 2404.97 | 4.08 ± 1.02 | 16.48 | 3037.3 | 19.0 |
| ort-cpu | int8 | cpu | _TODO(vijay): Phase 3_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| — | int4 | cpu | _TODO(vijay): Phase 3_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| ort-qnn | int8 | Snapdragon NPU | _TODO(vijay): run on Snapdragon (Phase 5)_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

**Reading the FP32 baseline honestly:**
- **Identical perplexity (19.0)** for PyTorch and ONNX Runtime confirms the ONNX export is numerically faithful — a key correctness check before quantizing.
- **Size:** the PyTorch number is the model's on-disk **bf16** `safetensors` as distributed by the hub (~942 MB); the ONNX export is **fp32** (~2405 MB ≈ 0.5B params × 4 bytes). Different dtypes, so this is the one non-apples-to-apples cell — quantization (Phase 3) is where size drops meaningfully.
- **Peak RAM** is sampled from process RSS during the measured runs (`psutil`).

Reproduce: `edgellm export && edgellm benchmark` (writes `results/benchmarks.json` + `results/benchmark.md`).

---

## Setup

This project targets **Python 3.10–3.12**. (On this machine the default `python3` is 3.14, which the ML stack does not yet fully support, so the venv is built on Python 3.11.)

```bash
# From the repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"          # core + lint/test tooling
# Later phases: pip install -e ".[onnx,eval,report]"
```

## Usage

```bash
# Show the resolved model/device for this machine
edgellm info

# Generate text (downloads the model on first run)
edgellm generate --prompt "Explain what quantization is in one sentence."

# Deterministic / reproducible generation
edgellm generate -p "Write a haiku about edge AI." --greedy --max-new-tokens 48

# Export to ONNX (FP32) and run the full baseline benchmark
edgellm export
edgellm benchmark              # add --skip-ppl to skip perplexity
```

## Development

```bash
ruff check .        # lint
black --check .     # format check
pytest              # tests
```

CI (GitHub Actions) runs all three on every push.

## License

[MIT](LICENSE) © 2026 Vijay Kapse
