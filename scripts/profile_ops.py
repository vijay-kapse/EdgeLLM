"""Per-operator time for quantcost's decode loop, split into prefill and decode.

Runs the same OnnxLiteRunner.generate() the benchmark uses, with ONNX Runtime's
profiler on, then attributes every kernel event to the model run that contains
it. The first run of each generate() is the prefill; the rest are single-token
decode steps. The first generate() is a warmup and is dropped.

    python scripts/profile_ops.py --precisions fp32,int8 --out profiles/
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path


def profile(model: str, precision: str, gen_tokens: int, runs: int, out: Path) -> dict:
    import onnxruntime as ort

    from edgellm.config import GenerationConfig
    from edgellm.hub import resolve
    from edgellm.ort_lite import OnnxLiteRunner
    from edgellm.sweep import BENCH_PROMPT

    real_session = ort.InferenceSession

    def profiled_session(path, opts=None, **kw):
        opts = opts or ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = str(out / f"ort-{precision}")
        return real_session(path, opts, **kw)

    ort.InferenceSession = profiled_session
    try:
        resolved = resolve(model, precision)
        runner = OnnxLiteRunner(str(resolved.onnx_path), str(resolved.tokenizer_path))
    finally:
        ort.InferenceSession = real_session

    cfg = GenerationConfig(max_new_tokens=gen_tokens, do_sample=False)
    for _ in range(runs + 1):
        runner.generate(BENCH_PROMPT, cfg)
    trace = Path(runner.session.end_profiling())

    events = json.loads(trace.read_text())
    model_runs = sorted((e for e in events if e.get("name") == "model_run"), key=lambda e: e["ts"])
    kernels = [e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]

    # model_run index -> phase; drop the warmup generate().
    phase = {}
    for i in range(len(model_runs)):
        g, step = divmod(i, gen_tokens)
        if g == 0:
            continue
        phase[i] = "prefill" if step == 0 else "decode"

    starts = [r["ts"] for r in model_runs]
    totals = {"prefill": collections.Counter(), "decode": collections.Counter()}
    for k in kernels:
        i = bisect.bisect_right(starts, k["ts"]) - 1
        if i in phase:
            totals[phase[i]][k["args"].get("op_name", "?")] += k["dur"]

    run_us = {"prefill": 0, "decode": 0}
    for i, r in enumerate(model_runs):
        if i in phase:
            run_us[phase[i]] += r["dur"]

    decode_steps = runs * (gen_tokens - 1)
    return {
        "precision": precision,
        "ort": ort.__version__,
        "threads": runner.intra_op_threads,
        "decode_steps": decode_steps,
        "decode_ms_per_token": run_us["decode"] / decode_steps / 1000,
        "prefill_ms": run_us["prefill"] / runs / 1000,
        "decode_ops_us_per_token": {
            op: us / decode_steps for op, us in totals["decode"].most_common()
        },
        "prefill_ops_us": {op: us / runs for op, us in totals["prefill"].most_common()},
        "trace": trace.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--precisions", default="fp32,int8")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("profiles"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = [
        profile(args.model, p, args.gen_tokens, args.runs, args.out)
        for p in args.precisions.split(",")
    ]
    (args.out / "summary.json").write_text(json.dumps(results, indent=1))

    for r in results:
        print(f"\n#### {r['precision']} · ORT {r['ort']} · {r['threads']} threads")
        print(
            f"decode {r['decode_ms_per_token']:.2f} ms/token · prefill {r['prefill_ms']:.1f} ms\n"
        )
        print("| decode op | µs/token | share |")
        print("|---|---|---|")
        ops = r["decode_ops_us_per_token"]
        total = sum(ops.values()) or 1
        for op, us in list(ops.items())[:12]:
            print(f"| {op} | {us:.0f} | {100 * us / total:.1f}% |")


if __name__ == "__main__":
    main()
