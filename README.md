# edge-llm-bench

**Find out what quantization actually costs you — on your machine, not someone else's.**

[![CI](https://github.com/vijay-kapse/EdgeLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/vijay-kapse/EdgeLLM/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/edge-llm-bench)](https://pypi.org/project/edge-llm-bench/)
[![Python](https://img.shields.io/pypi/pyversions/edge-llm-bench)](https://pypi.org/project/edge-llm-bench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```bash
pip install edge-llm-bench
edge-llm-bench run
```

That's it. Two to three minutes later you get speed, size, memory and *quality*
for fp32 / int8 / int4 on your own CPU, with a plain-language verdict.

No PyTorch. No GPU. No quantizing anything yourself — the pre-quantized ONNX
already published on the Hugging Face Hub gets downloaded and measured, so the
whole install is ONNX Runtime, NumPy and a tokenizer, and finishes in about ten
seconds.

---

## The thing nobody tells you about int4

"Quantize it, it'll be smaller and faster" is half right. Here is a real run on
an Apple M4:

| Precision | Size (MB) | Throughput (tok/s) | Speedup | Peak RAM (MB) | Perplexity | PPL change |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 515 | 64.54 | baseline | 1067 | 23.07 | baseline |
| int8 | 131 | 20.83 | **0.32x** | 819 | 25.02 | +8.5% |
| q4 | 174 | 12.26 | **0.19x** | 716 | 28.35 | +22.9% |

Both quantized models are 3–4x smaller and **3–5x slower**. That is not a bug in
the export and it is not unique to this chip: when ONNX Runtime has no native
kernel for a quantized format on your hardware, it dequantizes back to float
*inside* the matmul. You pay the unpacking cost on every single token and keep
none of the arithmetic savings.

So the honest answer to "should I quantize?" is **it depends entirely on your
hardware**, which is exactly why a benchmark you run yourself beats a number
from someone's blog post. On this machine you quantize to fit in memory, not to
go fast. On yours, it might be the opposite — that is the thing worth finding out.

**[See what other machines measured →](results/LEADERBOARD.md)**

## Add your machine

```bash
edge-llm-bench run
edge-llm-bench submit
```

`submit` validates your result, forks this repo, commits the card and opens the
pull request for you. No GitHub CLI? It prints a prefilled link instead.

Unusual hardware is the most valuable kind: Raspberry Pi, old ThinkPads,
Snapdragon laptops, bare-metal ARM. The interesting question is not who has the
fastest laptop — it is *where quantization pays off and where it backfires*, and
that only becomes visible across many real machines.

See [CONTRIBUTING.md](CONTRIBUTING.md) for what makes a good submission and
exactly what a card contains (no hostname, no username, no paths — the
fingerprint is one auditable function).

## Usage

```bash
# A different model — anything with ONNX on the Hub works
edge-llm-bench run --model onnx-community/Qwen2.5-0.5B-Instruct

# Which precisions does a model actually publish?
edge-llm-bench models --model onnx-community/Qwen2.5-0.5B-Instruct

# Speed and size only; skips the perplexity pass and is much faster
edge-llm-bench run --precisions fp32,int8 --skip-perplexity

# Pin threads to compare against a specific configuration
edge-llm-bench run --threads 4

# Rebuild the leaderboard from every submitted card
edge-llm-bench leaderboard
```

Any Hub repo following the `onnx-community` / `transformers.js` layout works
unchanged — that is thousands of models, including everything under
[onnx-community](https://huggingface.co/onnx-community).

## How the numbers are produced

The headline claim of this project is that its numbers are real, so the
methodology is worth stating plainly.

**Every precision is measured in its own subprocess.** Peak RSS is a per-process
high-water mark and an ONNX Runtime session does not return all of its arenas
when dropped, so measuring several precisions in one process reports the *union*
of their footprints and blames whichever ran last. Isolation is the only way the
peak-RAM column means what it says.

**The file cache is warmed before anything is timed.** ONNX Runtime mmaps
weights and faults them in lazily. Measured cold — straight after the download —
an fp32 baseline came in at 17 tok/s; warm, the same machine and build measured
64. A 4x error on the number every other row is divided by is not a rounding
detail, so the model file is read through once before the session is built.

**Throughput is total tokens over total time**, derived from the same mean
latency that is reported, never the average of each run's own rate. Those are
different statistics (ratio of means vs. mean of ratios) and they disagree by a
few percent, which would make two columns of the same card contradict each other.

**Decoding is greedy and fixed-length.** No sampling, no early EOS stop, so every
precision does exactly the same amount of work and throughput does not depend on
the RNG.

**Perplexity is scored against a corpus pinned inside the package** — a fixed
65,342-byte slice of WikiText-2, hash-checked on load and recorded in every card.
A leaderboard where machines score different text is not a leaderboard, and this
also removes a heavy `datasets` dependency. Method: non-overlapping 512-token
windows, summed next-token cross-entropy, `exp(total_nll / total_tokens)`. The
NumPy implementation is tested against the uniform-distribution case, where the
right answer is analytically the vocabulary size, and agrees with PyTorch's
`cross_entropy` to 4e-07 relative.

**Thread count is always pinned** to physical cores (SMT siblings contend for the
same vector units) and recorded, because leaving it at ORT's default stores "0 —
decide for me" and makes two very different runs look identical.

**Instability is reported, not hidden.** If run-to-run latency varies by more
than 15%, the report says the machine was too busy and asks you to re-run.

### What validation can and cannot do

CI rejects cards that are malformed, internally inconsistent, scored against a
modified corpus, run with an unpinned thread count, or carrying identifying
information. It **cannot** prove a number came from real silicon — nothing short
of attested execution could. The defence is that every input is pinned, so a
fabricated card must be self-consistent across all of them, and anyone can re-run
the exact configuration recorded in the card. Cards are reviewed, not trusted.

## Authoring quantized artifacts yourself

Everything above *consumes* pre-quantized ONNX. The original project also
*produces* it — export, quantize, a C++ inference harness, a SIMD INT8 GEMM
kernel, an Android app and a Qualcomm Snapdragon NPU path. That path needs the
heavy dependencies:

```bash
pip install -e ".[quantize]"
edgellm export --help
edgellm quantize --help
edgellm bench --help      # the original torch + Optimum harness
```

See [docs/AUTHORING.md](docs/AUTHORING.md) for the full pipeline, the C++ harness
and the Snapdragon notes.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

CI asserts that importing the CLI pulls in no heavy dependency. If you need
torch, import it inside the function that uses it — the fast install is what
makes a one-command benchmark viable for someone who has never heard of this
project.

## License

MIT — see [LICENSE](LICENSE). The bundled eval corpus is WikiText-2, CC BY-SA 4.0;
see [edgellm/data/SOURCE.md](edgellm/data/SOURCE.md).
