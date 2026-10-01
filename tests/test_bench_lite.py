"""Tests for the lightweight path: corpus integrity, NumPy scoring, cards, rendering.

Nothing here touches the network or loads a real model: the ONNX session is stood
in for by a fake runner with a known logit distribution, so the perplexity math is
checked against a value that can be computed by hand.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from edgellm.card import Fingerprint, PrecisionRow, ResultCard, slugify
from edgellm.eval_lite import (
    EVAL_CORPUS_SHA256,
    _log_softmax_gather,
    evaluate_perplexity,
    load_corpus,
)
from edgellm.hub import DEFAULT_PRECISIONS, PRECISION_FILES, ModelResolutionError, resolve
from edgellm.leaderboard import build_leaderboard, render_leaderboard_markdown
from edgellm.render import tradeoff_table, verdict
from edgellm.sweep import prompt_sha256

# --------------------------------------------------------------- eval corpus


def test_bundled_corpus_matches_its_recorded_hash():
    """If this fails the package is corrupt and no perplexity it reports is comparable."""
    assert load_corpus()
    import hashlib

    assert hashlib.sha256(load_corpus().encode()).hexdigest() == EVAL_CORPUS_SHA256


def test_corpus_is_substantial_enough_for_the_default_windows():
    # 8 windows x 512 tokens needs well over 4096 tokens of text.
    assert len(load_corpus()) > 50_000


def test_log_softmax_gather_matches_hand_computed_value():
    """Uniform logits: every token is equally likely, so NLL per token is log(V)."""
    vocab = 8
    logits = np.zeros((5, vocab), dtype=np.float32)
    labels = np.array([0, 1, 2, 3, 4])
    assert _log_softmax_gather(logits, labels) == pytest.approx(5 * math.log(vocab), rel=1e-9)


def test_log_softmax_gather_is_stable_for_large_logits():
    """A naive exp() would overflow here; the max-subtraction must hold."""
    logits = np.full((3, 4), 10_000.0, dtype=np.float32)
    labels = np.array([0, 1, 2])
    result = _log_softmax_gather(logits, labels)
    assert math.isfinite(result)
    assert result == pytest.approx(3 * math.log(4), rel=1e-6)


def test_log_softmax_gather_chunking_does_not_change_the_result():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(40, 32)).astype(np.float32) * 5
    labels = rng.integers(0, 32, size=40)
    assert _log_softmax_gather(logits, labels, chunk=7) == pytest.approx(
        _log_softmax_gather(logits, labels, chunk=1000), rel=1e-9
    )


class _UniformRunner:
    """Predicts a uniform distribution, so perplexity must equal the vocab size."""

    def __init__(self, vocab: int = 64):
        self.vocab_size = vocab

    def forward_logits(self, input_ids: np.ndarray) -> np.ndarray:
        batch, seq = input_ids.shape
        return np.zeros((batch, seq, self.vocab_size), dtype=np.float32)


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeTokenizer:
    """Emits ids cycling within ``vocab``, the way a real tokenizer always does."""

    def __init__(self, n_tokens: int = 4096, vocab: int = 64):
        self.n_tokens = n_tokens
        self.vocab = vocab

    def encode(self, _text: str) -> _Encoding:
        return _Encoding([i % self.vocab for i in range(self.n_tokens)])


def test_perplexity_of_a_uniform_model_is_the_vocab_size():
    """The one case where the correct answer is known analytically."""
    result = evaluate_perplexity(_UniformRunner(64), _FakeTokenizer(2048, vocab=64), windows=2)
    assert result.perplexity == pytest.approx(64.0, rel=1e-4)
    assert result.corpus_sha256 == EVAL_CORPUS_SHA256


def test_perplexity_stops_cleanly_when_the_corpus_runs_out():
    """Asking for more windows than there are tokens reports what was scored."""
    result = evaluate_perplexity(_UniformRunner(16), _FakeTokenizer(600, vocab=16), windows=8)
    assert result.windows == 2  # 512 + 88
    assert result.scored_tokens == 598


# ---------------------------------------------------------------------- hub


def test_default_precisions_are_all_known():
    assert set(DEFAULT_PRECISIONS) <= set(PRECISION_FILES)


def test_fp32_is_the_first_default_precision():
    """Everything else is reported relative to it, so it must be measured."""
    assert DEFAULT_PRECISIONS[0] == "fp32"


def test_unknown_precision_raises_before_any_network_call():
    with pytest.raises(ModelResolutionError, match="Unknown precision"):
        resolve("some/model", "int3")


# --------------------------------------------------------------------- card


def test_slugify_is_filesystem_safe():
    assert slugify("Apple M4 Pro (10-core)") == "apple-m4-pro-10-core"
    assert slugify("///") == "unknown"


def _fingerprint(**kw) -> Fingerprint:
    base = dict(
        cpu="Apple M4",
        arch="arm64",
        physical_cores=10,
        logical_cores=10,
        ram_gb=16.0,
        os="Darwin",
        os_release="25.5.0",
        python="3.11.15",
        onnxruntime="1.27.0",
        provider="CPUExecutionProvider",
        intra_op_threads=10,
    )
    base.update(kw)
    return Fingerprint(**base)


def test_fingerprint_carries_no_identifying_fields():
    """Cards are committed to a public repo; this is the privacy guarantee."""
    fields = set(_fingerprint().__dict__)
    forbidden = {"hostname", "user", "username", "mac", "ip", "serial", "machine_id", "path"}
    assert fields & forbidden == set()


def _card(rows: list[PrecisionRow]) -> ResultCard:
    return ResultCard(
        schema_version=1,
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="main",
        created_utc="2026-10-01T00:00:00Z",
        tool_version="0.2.0",
        machine=_fingerprint(),
        rows=rows,
        eval_windows=8,
        eval_corpus_sha256=EVAL_CORPUS_SHA256,
        warmup_runs=1,
        gen_tokens=32,
        prompt_sha256=prompt_sha256(),
    )


def _rows() -> list[PrecisionRow]:
    def row(precision, size, tps, ppl, ram):
        return PrecisionRow(
            precision=precision,
            size_mb=size,
            latency_s_mean=round(32 / tps, 4),
            latency_s_std=0.01,
            tokens_per_second=tps,
            peak_ram_mb=ram,
            perplexity=ppl,
            generated_tokens=32,
            measured_runs=3,
        )

    return [
        row("fp32", 515.0, 17.52, 23.07, 950.0),
        row("int8", 131.0, 15.53, 25.02, 813.0),
        row("q4", 174.0, 20.29, 28.35, 693.0),
    ]


def test_card_filename_is_stable_so_a_rerun_updates_its_own_row():
    card = _card(_rows())
    assert card.filename == "apple-m4-darwin--smollm2-135m-instruct.json"
    assert _card(_rows()).filename == card.filename


def test_speedup_vs_baseline():
    speedups = _card(_rows()).speedup_vs("fp32")
    assert speedups["q4"] == pytest.approx(20.29 / 17.52, rel=1e-3)
    assert speedups["int8"] < 1.0  # slower than fp32 on this hardware


# ------------------------------------------------------------------- render


def test_tradeoff_table_marks_the_baseline_and_formats_deltas():
    table = tradeoff_table(_card(_rows()))
    assert "| fp32 |" in table and "baseline" in table
    assert "+8.5%" in table  # int8 perplexity delta, one decimal and signed
    assert "0.89x" in table  # int8 is slower


def test_verdict_calls_out_a_precision_that_is_slower_than_baseline():
    """The finding people most need and least expect must be stated outright."""
    findings = "\n".join(verdict(_card(_rows())))
    assert "slower than fp32" in findings
    assert "fit in memory, not to go faster" in findings


def test_verdict_reports_the_fastest_precision():
    assert "Best throughput" in "\n".join(verdict(_card(_rows())))


def test_verdict_without_a_baseline_says_so_instead_of_guessing():
    rows = [r for r in _rows() if r.precision != "fp32"]
    findings = verdict(_card(rows))
    assert "No fp32 baseline" in findings[0]


# -------------------------------------------------------------- leaderboard


def test_leaderboard_aggregates_and_counts_slower_machines(tmp_path):
    card = _card(_rows())
    path = tmp_path / card.filename
    path.write_text(json.dumps(card.as_dict(), indent=2))

    board = build_leaderboard([path])
    assert not board["rejected"]
    assert board["machine_count"] == 1

    by_precision = {r["precision"]: r for r in board["aggregate"]["by_model_precision"]}
    assert by_precision["int8"]["slower_than_baseline"] == 1
    assert by_precision["q4"]["slower_than_baseline"] == 0
    assert by_precision["q4"]["speedup_median"] == pytest.approx(1.158, abs=1e-3)

    markdown = render_leaderboard_markdown(board)
    assert "Apple M4" in markdown
    assert "Does quantization pay off?" in markdown


def test_leaderboard_quarantines_an_invalid_card(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "rows": []}))
    board = build_leaderboard([bad])
    assert board["entries"] == []
    assert board["rejected"][0]["file"] == "bad.json"


def test_noncomparable_cards_are_listed_but_excluded_from_the_aggregate(tmp_path):
    card = _card(_rows())
    payload = card.as_dict()
    payload["gen_tokens"] = 64
    for row in payload["rows"]:
        row["generated_tokens"] = 64
        row["latency_s_mean"] = round(64 / row["tokens_per_second"], 4)

    path = tmp_path / "other.json"
    path.write_text(json.dumps(payload))

    board = build_leaderboard([path])
    assert len(board["entries"]) == 1
    assert board["entries"][0]["comparable"] is False
    assert board["aggregate"]["by_model_precision"] == []


def test_mismatched_tokenizer_vocab_raises_a_clear_error():
    """A stale tokenizer.json must not surface as a bare IndexError."""
    with pytest.raises(ValueError, match="does not match its ONNX graph"):
        evaluate_perplexity(_UniformRunner(16), _FakeTokenizer(600, vocab=999), windows=1)
