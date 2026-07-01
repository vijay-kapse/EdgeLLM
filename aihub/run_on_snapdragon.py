"""Compile + profile a quantized ONNX model on a real Qualcomm Snapdragon device.

This uses **Qualcomm AI Hub** (https://aihub.qualcomm.com) to compile the model
to the QNN runtime and run a profiling job on physical hardware in Qualcomm's
device farm, returning true on-device latency/throughput.

Running it needs a *free* AI Hub API token (see `check_auth()` for the exact
steps). Without a token the script prints those steps and exits cleanly — it
never fabricates device numbers. When it does run, it appends a real
`ort-qnn` row to `results/benchmarks.json`.

Usage:
    python aihub/run_on_snapdragon.py \
        --model artifacts/onnx/Qwen__Qwen2.5-0.5B-Instruct-int8-dynamic \
        --device "Snapdragon 8 Elite QRD" --seq 64
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TOKEN_ENV = "QAI_HUB_API_TOKEN"
CLIENT_INI = Path.home() / ".qai_hub" / "client.ini"


def check_auth() -> bool:
    """Return True if an AI Hub token is configured; else print setup steps."""
    if os.environ.get(TOKEN_ENV) or CLIENT_INI.exists():
        return True
    print(
        "\n".join(
            [
                "Qualcomm AI Hub token not found. To enable the on-device NPU run:",
                "  1. Create a free account at https://aihub.qualcomm.com",
                "  2. Copy your API token from https://aihub.qualcomm.com/account/",
                "  3. Configure it one of two ways:",
                "       pip install qai-hub",
                "       qai-hub configure --api_token <YOUR_TOKEN>",
                f"     (or export {TOKEN_ENV}=<YOUR_TOKEN>)",
                "  4. Re-run this script.",
                "",
                "Until then the Snapdragon NPU row stays a labeled TODO placeholder.",
            ]
        )
    )
    return False


def fixed_input_specs(onnx_path: Path, seq: int, past: int) -> dict:
    """Derive fixed-shape input specs from the ONNX model's dynamic inputs.

    AI Hub compilation needs concrete shapes. Dynamic dims are resolved as:
    batch -> 1, sequence_length -> ``seq``, past_sequence_length -> ``past``.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    specs: dict[str, tuple] = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                text = str(dim)
                if "past" in text:
                    shape.append(past)
                elif "batch" in text:
                    shape.append(1)
                elif "sequence" in text:
                    shape.append(seq)
                else:
                    shape.append(1)
        dtype = "int64" if "int" in inp.type else "float32"
        specs[inp.name] = (tuple(shape), dtype)
    return specs


class SnapdragonProfiler:
    """Drive an AI Hub compile + profile job for one ONNX model."""

    def __init__(self, model_dir: Path, device_name: str, seq: int, past: int) -> None:
        self.model_dir = model_dir
        self.device_name = device_name
        self.seq = seq
        self.past = past

    def run(self, results_json: Path) -> dict:
        import qai_hub as hub

        onnx_path = self.model_dir / "model.onnx"
        devices = hub.get_devices(self.device_name)
        if not devices:
            raise SystemExit(f"No AI Hub device matches '{self.device_name}'.")
        device = devices[0]
        print(f"Target device: {device.name}")

        specs = fixed_input_specs(onnx_path, self.seq, self.past)
        print(f"Input specs: { {k: v[0] for k, v in specs.items()} }")

        # Compile to the QNN runtime, then profile on the physical device.
        compile_job = hub.submit_compile_job(
            model=str(onnx_path),
            device=device,
            input_specs=specs,
            options="--target_runtime qnn_context_binary",
        )
        target_model = compile_job.get_target_model()
        profile_job = hub.submit_profile_job(model=target_model, device=device)
        profile = profile_job.download_profile()

        # Extract the mean inference latency (microseconds) from the profile.
        exec_us = _mean_exec_us(profile)
        latency_s = exec_us / 1e6
        row = {
            "backend": "ort-qnn",
            "precision": _precision_from_dir(self.model_dir),
            "device": device.name,
            "model_id": self.model_dir.name,
            "size_mb": None,
            "latency_s_mean": round(latency_s, 6),
            "latency_s_std": 0.0,
            "tokens_per_second": round(1.0 / latency_s, 2) if latency_s > 0 else 0.0,
            "peak_ram_mb": None,
            "perplexity": None,
            "generated_tokens": 1,
            "measured_runs": 1,
        }
        _merge_row(results_json, row)
        print(f"On-device latency: {latency_s * 1000:.3f} ms  ->  wrote {results_json}")
        return row


def _mean_exec_us(profile: dict) -> float:
    """Pull mean execution time (us) out of an AI Hub profile dict."""
    ex = profile.get("execution_summary", {})
    for key in ("estimated_inference_time", "execution_time"):
        if key in ex:
            return float(ex[key])
    raise KeyError("could not find execution time in profile summary")


def _precision_from_dir(model_dir: Path) -> str:
    name = model_dir.name
    if "int4" in name:
        return "int4"
    if "int8" in name:
        return "int8"
    return "fp32"


def _merge_row(results_json: Path, row: dict) -> None:
    results_json.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(results_json.read_text()) if results_json.exists() else []
    rows = [
        r
        for r in rows
        if not (r["backend"] == row["backend"] and r["precision"] == row["precision"])
    ]
    rows.append(row)
    results_json.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="ONNX model directory.")
    parser.add_argument("--device", default="Snapdragon 8 Elite QRD", help="AI Hub device name.")
    parser.add_argument("--seq", type=int, default=64, help="Fixed sequence length for compile.")
    parser.add_argument("--past", type=int, default=0, help="Fixed past-sequence length.")
    parser.add_argument("--results", type=Path, default=Path("results/benchmarks.json"))
    args = parser.parse_args()

    if not check_auth():
        raise SystemExit(0)

    SnapdragonProfiler(args.model, args.device, args.seq, args.past).run(args.results)


if __name__ == "__main__":
    main()
