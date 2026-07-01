"""Tests for report chart rendering and INT4 availability reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgellm.quantize import Quantizer


def test_int4_availability_keys() -> None:
    avail = Quantizer.int4_availability()
    assert "onnxruntime block-wise INT4" in avail
    # ORT INT4 always runs on CPU; the value should say so.
    assert "available" in avail["onnxruntime block-wise INT4"]


def test_render_charts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from edgellm.report import render_charts

    rows = [
        {
            "backend": "ort-cpu",
            "precision": "int8",
            "device": "cpu",
            "size_mb": 605.0,
            "tokens_per_second": 38.3,
            "perplexity": 20.1,
        },
        {
            "backend": "pytorch",
            "precision": "int8",
            "device": "cpu",
            "size_mb": None,  # exercises the n/a path
            "tokens_per_second": 15.9,
            "perplexity": 58.4,
        },
    ]
    json_path = tmp_path / "b.json"
    json_path.write_text(json.dumps(rows))
    out = render_charts(json_path, tmp_path / "chart.png")
    assert out.exists() and out.stat().st_size > 0
