"""Tests for card validation — the gate that keeps the leaderboard meaningful.

These matter more than most tests here: the validator is the only thing standing
between a public leaderboard and numbers that cannot be compared. Each test below
is a submission that *should* be caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgellm.card import CARD_SCHEMA_VERSION
from edgellm.eval_lite import EVAL_CORPUS_SHA256
from edgellm.sweep import prompt_sha256
from edgellm.validate import validate_card


def _row(precision="fp32", *, tps=20.0, gen=32, ppl=23.0, latency=1.6):
    return {
        "precision": precision,
        "size_mb": 515.0,
        "latency_s_mean": latency,
        "latency_s_std": 0.01,
        "tokens_per_second": tps,
        "peak_ram_mb": 950.0,
        "perplexity": ppl,
        "generated_tokens": gen,
        "measured_runs": 3,
    }


def _card(**overrides):
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "main",
        "created_utc": "2026-10-01T00:00:00Z",
        "tool_version": "0.2.0",
        "machine": {
            "cpu": "Apple M4",
            "arch": "arm64",
            "physical_cores": 10,
            "logical_cores": 10,
            "ram_gb": 16.0,
            "os": "Darwin",
            "os_release": "25.5.0",
            "python": "3.11.15",
            "onnxruntime": "1.27.0",
            "provider": "CPUExecutionProvider",
            "intra_op_threads": 4,
            "thread_policy": "macos-perflevel0",
        },
        # 32 tokens / 1.6 s = 20 tok/s, so the derived check is satisfied.
        "rows": [_row("fp32"), _row("int8", tps=16.0, latency=2.0, ppl=25.0)],
        "eval_windows": 8,
        "eval_corpus_sha256": EVAL_CORPUS_SHA256,
        "warmup_runs": 1,
        "gen_tokens": 32,
        "prompt_sha256": prompt_sha256(),
        "notes": "",
        "submitted_by": "",
        "extra": {},
    }
    card.update(overrides)
    return card


def write(tmp_path: Path, card: dict, name: str = "card.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(card))
    return path


def test_valid_card_passes_and_is_comparable(tmp_path):
    report = validate_card(write(tmp_path, _card()))
    assert report.ok, report.errors
    assert report.comparable


def test_unreadable_json_is_reported_not_raised(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    report = validate_card(path)
    assert not report.ok
    assert "unreadable JSON" in report.errors[0]


def test_throughput_must_agree_with_latency(tmp_path):
    """The headline number cannot be edited without also editing its inputs."""
    card = _card(rows=[_row("fp32", tps=400.0)])  # 32/1.6 = 20, not 400
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("disagrees with" in e for e in report.errors)


def test_mismatched_eval_corpus_is_rejected(tmp_path):
    card = _card(eval_corpus_sha256="0" * 64)
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("eval_corpus_sha256" in e for e in report.errors)


def test_mismatched_prompt_is_rejected(tmp_path):
    card = _card(prompt_sha256="0" * 64)
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("prompt_sha256" in e for e in report.errors)


def test_unpinned_thread_count_is_rejected(tmp_path):
    machine = dict(_card()["machine"], intra_op_threads=0)
    report = validate_card(write(tmp_path, _card(machine=machine)))
    assert not report.ok
    assert any("intra_op_threads" in e for e in report.errors)


def test_identifying_fields_are_rejected(tmp_path):
    machine = dict(_card()["machine"], hostname="vijays-laptop")
    report = validate_card(write(tmp_path, _card(machine=machine)))
    assert not report.ok
    assert any("identifying information" in e for e in report.errors)


def test_generated_tokens_must_match_card_budget(tmp_path):
    card = _card(rows=[_row("fp32", gen=64, latency=3.2)])  # 64/3.2 = 20 tok/s, consistent
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("does not match" in e for e in report.errors)


def test_duplicate_precision_is_rejected(tmp_path):
    card = _card(rows=[_row("fp32"), _row("fp32")])
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("duplicate precision" in e for e in report.errors)


def test_unknown_precision_is_rejected(tmp_path):
    card = _card(rows=[_row("int3")])
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("unknown precision" in e for e in report.errors)


def test_implausible_throughput_is_rejected(tmp_path):
    card = _card(rows=[_row("fp32", tps=500_000.0, latency=32 / 500_000.0)])
    report = validate_card(write(tmp_path, card))
    assert not report.ok
    assert any("plausible" in e for e in report.errors)


def test_old_schema_version_is_rejected(tmp_path):
    report = validate_card(write(tmp_path, _card(schema_version=CARD_SCHEMA_VERSION - 1)))
    assert not report.ok
    assert any("schema_version" in e for e in report.errors)


def test_nondefault_settings_pass_but_are_not_comparable(tmp_path):
    """A 64-token run is valid and worth listing, but must not be ranked."""
    card = _card(gen_tokens=64, rows=[_row("fp32", gen=64, latency=3.2)])
    report = validate_card(write(tmp_path, card))
    assert report.ok, report.errors
    assert not report.comparable


def test_skipped_perplexity_does_not_require_corpus_hash(tmp_path):
    rows = [_row("fp32", ppl=None), _row("int8", tps=16.0, latency=2.0, ppl=None)]
    card = _card(rows=rows, eval_windows=0, eval_corpus_sha256="")
    report = validate_card(write(tmp_path, card))
    assert report.ok, report.errors


@pytest.mark.parametrize("field", ["size_mb", "latency_s_mean", "tokens_per_second", "peak_ram_mb"])
def test_nonpositive_measurements_are_rejected(tmp_path, field):
    row = _row("fp32")
    row[field] = 0
    report = validate_card(write(tmp_path, _card(rows=[row])))
    assert not report.ok
