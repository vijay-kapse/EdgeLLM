"""Checks for a DQ->MatMulNBits change in ONNX Runtime.

python scripts/check_fusion.py logits --out a.npz     # teacher-forced logits for the int8 model
python scripts/check_fusion.py compare a.npz b.npz    # max |diff| and argmax agreement
python scripts/check_fusion.py load REPO FILE         # session loads; count fused ops
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
from pathlib import Path

import numpy as np

PROMPTS = [
    "Explain what neural network quantization is, in one short paragraph.",
    "The capital of France is",
    "def fibonacci(n):\n    ",
]


def logits(model: str, precision: str, out: Path, config: dict[str, str]) -> None:
    import onnxruntime as ort

    from edgellm.hub import resolve
    from edgellm.ort_lite import OnnxLiteRunner

    real_session = ort.InferenceSession

    def configured_session(path, opts=None, **kw):
        opts = opts or ort.SessionOptions()
        for key, value in config.items():
            opts.add_session_config_entry(key, value)
        return real_session(path, opts, **kw)

    ort.InferenceSession = configured_session
    try:
        resolved = resolve(model, precision)
        runner = OnnxLiteRunner(str(resolved.onnx_path), str(resolved.tokenizer_path))
    finally:
        ort.InferenceSession = real_session
    arrays = {}
    for i, prompt in enumerate(PROMPTS):
        ids = np.asarray([runner.tokenizer.encode(prompt).ids], dtype=np.int64)
        arrays[f"p{i}"] = runner.forward_logits(ids)[0]
    np.savez(out, ort=np.array(ort.__version__), **arrays)
    print(f"ORT {ort.__version__}: wrote {len(PROMPTS)} prompts to {out}")


def compare(a: Path, b: Path) -> int:
    x, y = np.load(a), np.load(b)
    print(f"A = ORT {x['ort']}, B = ORT {y['ort']}")
    worst = 0.0
    for key in sorted(k for k in x.files if k.startswith("p")):
        la, lb = x[key], y[key]
        diff = float(np.max(np.abs(la - lb)))
        scale = float(np.max(np.abs(la)))
        agree = float(np.mean(la.argmax(-1) == lb.argmax(-1)))
        worst = max(worst, diff / scale)
        rel = diff / scale
        n = la.shape[0]
        print(f"  {key}: tokens={n} max|diff|={diff:.3e} (rel {rel:.2e}) argmax={agree:.0%}")
    print(f"worst relative diff {worst:.2e}")
    return 0


def load(repo: str, filename: str) -> int:
    import onnx
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename)
    opt = Path(tempfile.mkdtemp()) / "opt.onnx"
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(opt)
    try:
        ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - report the failure, that is the point
        print(f"LOAD FAILED on ORT {ort.__version__}: {e}")
        return 1
    model = onnx.load(str(opt), load_external_data=False)
    ops = collections.Counter(n.op_type for n in model.graph.node)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.g is not None and attr.g.node:
                ops.update(f"{n.op_type} (subgraph)" for n in attr.g.node)
    keep = {k: v for k, v in ops.items() if "MatMul" in k or "Dequantize" in k}
    print(f"loaded {repo}/{filename} on ORT {ort.__version__}: {keep}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("logits")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--precision", default="int8")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", action="append", default=[], help="Session config key=value.")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    ld = sub.add_parser("load")
    ld.add_argument("repo")
    ld.add_argument("filename")
    args = ap.parse_args()
    if args.cmd == "logits":
        logits(args.model, args.precision, args.out, dict(c.split("=", 1) for c in args.config))
        return 0
    if args.cmd == "compare":
        return compare(args.a, args.b)
    return load(args.repo, args.filename)


if __name__ == "__main__":
    sys.exit(main())
