"""A torch-free ONNX Runtime decoder: greedy generation with a KV cache, in NumPy.

Why this exists instead of ``optimum.onnxruntime``: Optimum is the right tool when
you are exporting and quantizing models, but it pulls in PyTorch and
``transformers``, which together are a multi-gigabyte install and the single
biggest reason a stranger abandons a benchmark run. The decode loop for a
causal LM is about a hundred lines of NumPy, so the default path implements it
directly and depends only on ``onnxruntime``, ``numpy`` and ``tokenizers``.

The graph contract (verified against the ``onnx-community`` / ``transformers.js``
exports, which is what the Hub publishes for thousands of models):

    inputs   input_ids                  int64  [batch, sequence]
             attention_mask             int64  [batch, total_sequence]
             position_ids               int64  [batch, sequence]
             past_key_values.{i}.key    T      [batch, kv_heads, past_sequence, head_dim]
             past_key_values.{i}.value  T      ... same
    outputs  logits                     float  [batch, sequence, vocab]
             present.{i}.key            T      [batch, kv_heads, total_sequence, head_dim]
             present.{i}.value          T      ... same

Layer count, KV-head count, head dim and the cache dtype are all read off the
session rather than from ``config.json``, so the runner is model-agnostic and
works for fp16 graphs (whose cache is float16) without special-casing.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from edgellm.base import GenerationResult, InferenceRunner
from edgellm.config import GenerationConfig

log = logging.getLogger(__name__)

_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(bfloat16)": np.float32,  # ORT has no bf16 numpy view; feed fp32 zeros.
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

_PAST_RE = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")


class OnnxLiteRunner(InferenceRunner):
    """Greedy ONNX Runtime generation with a KV cache, implemented in NumPy."""

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str,
        *,
        provider: str = "CPUExecutionProvider",
        name: str = "ort-lite",
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"Execution provider '{provider}' is not available in this ONNX "
                f"Runtime build (have: {', '.join(available)})."
            )

        # Thread count is the single largest lever on CPU throughput, so it is
        # always pinned to a concrete number and recorded in the result card.
        # Leaving it at ORT's default would store 0 ("decide for me"), which
        # makes two cards look identical while having run very differently.
        # Physical cores, not logical: SMT siblings contend for the same vector
        # units and consistently measure worse for GEMM-bound decode.
        if intra_op_threads is None:
            import psutil

            intra_op_threads = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.name = name
        self.provider = provider
        self.session = ort.InferenceSession(onnx_path, opts, providers=[provider])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.intra_op_threads = opts.intra_op_num_threads

        self._inspect_graph()

    # ------------------------------------------------------------------ setup

    def _inspect_graph(self) -> None:
        """Read layer count, KV geometry and cache dtype off the graph itself."""
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.input_names = set(inputs)

        layers = sorted(
            int(m.group(1))
            for name in inputs
            if (m := _PAST_RE.match(name)) and m.group(2) == "key"
        )
        if not layers:
            raise ValueError(
                "This ONNX graph exposes no 'past_key_values.*' inputs, so it is not a "
                "KV-cached decoder export. Use a model from onnx-community or any repo "
                "exported with Optimum's `--task text-generation-with-past`."
            )
        if layers != list(range(len(layers))):
            raise ValueError(f"Non-contiguous KV cache layer indices: {layers}")
        self.num_layers = len(layers)

        spec = inputs["past_key_values.0.key"]
        shape = spec.shape
        if len(shape) != 4 or not isinstance(shape[1], int) or not isinstance(shape[3], int):
            raise ValueError(
                f"Unexpected KV cache shape {shape}; expected "
                "[batch, kv_heads, past_sequence, head_dim] with static head dims."
            )
        self.kv_heads, self.head_dim = int(shape[1]), int(shape[3])
        self.cache_dtype = _ORT_TO_NUMPY.get(spec.type, np.float32)

        self.vocab_size = None
        for out in self.session.get_outputs():
            if out.name == "logits" and isinstance(out.shape[-1], int):
                self.vocab_size = int(out.shape[-1])

        self.uses_position_ids = "position_ids" in self.input_names

        log.debug(
            "graph: %d layers, %d kv heads, head_dim %d, cache %s, position_ids %s",
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            np.dtype(self.cache_dtype).name,
            self.uses_position_ids,
        )

    def _empty_cache(self, batch: int = 1) -> dict[str, np.ndarray]:
        """Zero-length KV cache: the graph's signal that this is the prefill step."""
        empty = np.zeros((batch, self.kv_heads, 0, self.head_dim), dtype=self.cache_dtype)
        feed = {}
        for i in range(self.num_layers):
            feed[f"past_key_values.{i}.key"] = empty
            feed[f"past_key_values.{i}.value"] = empty
        return feed

    # ------------------------------------------------------------- inference

    def _forward(
        self,
        input_ids: np.ndarray,
        past: dict[str, np.ndarray],
        past_len: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """One step (prefill or decode). Returns logits and the next cache."""
        batch, seq = input_ids.shape
        total = past_len + seq

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": np.ones((batch, total), dtype=np.int64),
            **past,
        }
        if self.uses_position_ids:
            feed["position_ids"] = np.arange(past_len, total, dtype=np.int64)[None, :].repeat(
                batch, axis=0
            )

        outputs = self.session.run(None, feed)
        names = [o.name for o in self.session.get_outputs()]
        by_name = dict(zip(names, outputs, strict=True))

        logits = by_name["logits"]
        next_past = {
            f"past_key_values.{i}.{kind}": by_name[f"present.{i}.{kind}"]
            for i in range(self.num_layers)
            for kind in ("key", "value")
        }
        return logits, next_past

    def forward_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Single full forward pass with no cache — what perplexity scoring needs."""
        logits, _ = self._forward(input_ids, self._empty_cache(input_ids.shape[0]), 0)
        return logits

    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        """Greedy decode ``max_new_tokens`` tokens, timing the whole loop.

        Decoding is always greedy here regardless of ``generation.temperature``:
        a benchmark needs a fixed token count and a deterministic path, and
        sampling would make throughput depend on the RNG. ``min_new_tokens`` is
        implied — generation never stops early on EOS, so every backend and
        precision is measured over exactly the same amount of work.
        """
        ids = self.tokenizer.encode(prompt).ids
        input_ids = np.asarray([ids], dtype=np.int64)
        prompt_tokens = len(ids)
        want = generation.max_new_tokens

        start = time.perf_counter()

        logits, past = self._forward(input_ids, self._empty_cache(), 0)
        past_len = prompt_tokens
        next_id = int(np.argmax(logits[0, -1]))
        generated = [next_id]

        for _ in range(want - 1):
            logits, past = self._forward(np.asarray([[next_id]], dtype=np.int64), past, past_len)
            past_len += 1
            next_id = int(np.argmax(logits[0, -1]))
            generated.append(next_id)

        latency_s = time.perf_counter() - start

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            backend=self.name,
            prompt=prompt,
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated),
            latency_s=latency_s,
            tokens_per_second=len(generated) / latency_s if latency_s > 0 else 0.0,
        )

    def close(self) -> None:
        self.session = None
