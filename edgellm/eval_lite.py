"""Perplexity on a pinned corpus, in NumPy — no torch, no ``datasets``.

Method (identical to :class:`edgellm.benchmark.PerplexityEvaluator`, so the heavy
and light paths produce comparable numbers): tokenize the corpus once, split it
into non-overlapping windows of ``max_length`` tokens, sum the next-token
cross-entropy over every window, and report ``exp(total_nll / total_tokens)``.

The corpus is the slice bundled at ``edgellm/data/eval_wikitext2.txt`` and its
hash is checked on load. That matters more than it looks: perplexity is only
comparable across machines if every machine scores byte-identical text, so a
submitted number carries the corpus hash and a mismatch is rejected rather than
quietly compared against everyone else.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from importlib import resources

import numpy as np

log = logging.getLogger(__name__)

#: sha256 of the bundled corpus. See edgellm/data/SOURCE.md for provenance.
EVAL_CORPUS_SHA256 = "34a2a43fcf49217f7a42133f961930d8c830ac7fe3cc7927f423f443083f97bf"

#: Scoring window, in tokens. Fixed because it changes the number it produces.
EVAL_WINDOW_TOKENS = 512

#: How many windows to score by default. 8 windows x 512 tokens is enough for a
#: stable number while keeping the eval to a few seconds on a laptop CPU.
DEFAULT_EVAL_WINDOWS = 8


class CorpusIntegrityError(RuntimeError):
    """The bundled evaluation corpus does not match its recorded hash."""


@dataclass(frozen=True)
class PerplexityResult:
    perplexity: float
    windows: int
    scored_tokens: int
    corpus_sha256: str


def load_corpus() -> str:
    """Read the bundled corpus and verify its hash."""
    text = (
        resources.files("edgellm").joinpath("data/eval_wikitext2.txt").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != EVAL_CORPUS_SHA256:
        raise CorpusIntegrityError(
            f"Bundled eval corpus hash mismatch: expected {EVAL_CORPUS_SHA256}, got {digest}. "
            "Reinstall the package; a modified corpus makes perplexity incomparable."
        )
    return text


def _log_softmax_gather(logits: np.ndarray, labels: np.ndarray, *, chunk: int = 64) -> float:
    """Summed negative log-likelihood of ``labels`` under ``logits``.

    Computed in row chunks with the max subtracted before ``exp``: a 512x152k
    logit block is ~300 MB in float32, and doing the whole thing at once is both
    a memory spike and a place where ``exp`` of a large logit overflows.
    """
    total = 0.0
    for start in range(0, logits.shape[0], chunk):
        block = logits[start : start + chunk].astype(np.float64, copy=False)
        rows = labels[start : start + chunk]
        shifted = block - block.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1)) + block.max(axis=-1)
        picked = block[np.arange(block.shape[0]), rows]
        total += float((logsumexp - picked).sum())
    return total


def evaluate_perplexity(
    runner,
    tokenizer,
    *,
    windows: int = DEFAULT_EVAL_WINDOWS,
    window_tokens: int = EVAL_WINDOW_TOKENS,
) -> PerplexityResult:
    """Score the bundled corpus with ``runner.forward_logits`` (NumPy in, NumPy out)."""
    text = load_corpus()
    ids = np.asarray(tokenizer.encode(text).ids, dtype=np.int64)

    # A repo whose tokenizer.json does not match its ONNX graph would otherwise
    # fail deep in the scoring loop with a bare IndexError.
    vocab = getattr(runner, "vocab_size", None)
    if vocab and int(ids.max()) >= vocab:
        raise ValueError(
            f"Tokenizer produced id {int(ids.max())} but the model's vocabulary is {vocab}. "
            "The tokenizer.json in this repo does not match its ONNX graph, so perplexity "
            "cannot be computed."
        )

    total_nll = 0.0
    total_tokens = 0
    scored_windows = 0

    for index in range(windows):
        start = index * window_tokens
        window = ids[start : start + window_tokens]
        if window.shape[0] < 2:
            break  # corpus exhausted; report what was actually scored

        logits = runner.forward_logits(window[None, :])
        nll = _log_softmax_gather(logits[0, :-1, :], window[1:])

        total_nll += nll
        total_tokens += int(window.shape[0] - 1)
        scored_windows += 1
        log.debug("window %d/%d: %d tokens", index + 1, windows, window.shape[0])

    if total_tokens == 0:
        raise ValueError("No tokens scored; corpus or window configuration is empty.")

    return PerplexityResult(
        perplexity=round(float(np.exp(total_nll / total_tokens)), 3),
        windows=scored_windows,
        scored_tokens=total_tokens,
        corpus_sha256=EVAL_CORPUS_SHA256,
    )
