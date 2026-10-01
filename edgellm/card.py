"""The result card: what one machine measured, in a shape that can be compared.

A card is the unit a contributor submits. It pairs the measured rows with enough
hardware and software context to make them meaningful, and with the knobs that
would otherwise silently change the numbers (thread count, token budget, eval
window count, corpus hash).

**On privacy.** Cards get committed to a public repository, so the fingerprint is
deliberately narrow: CPU model string, architecture, core count, rounded RAM, OS
name and release, and version strings. It never collects hostname, username,
local paths, MAC address, IP, serial number or any machine ID. ``fingerprint()``
is the only thing that reads the host, so that promise is auditable in one place.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Bumped when the card layout changes in a way a validator must notice.
CARD_SCHEMA_VERSION = 1


def _cpu_model() -> str:
    """A human-readable CPU model string, or 'unknown' if the OS will not say."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith(("model name", "Model")):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def _total_ram_gb() -> float:
    """Installed RAM, rounded to 1 GB — precise enough to compare, too coarse to identify."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        return 0.0


def slugify(text: str, *, max_length: int = 48) -> str:
    """Lowercase, hyphenated, filesystem-safe — used to name the card file."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "unknown"


@dataclass
class Fingerprint:
    """Non-identifying description of the machine and software stack."""

    cpu: str
    arch: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    os: str
    os_release: str
    python: str
    onnxruntime: str
    provider: str
    intra_op_threads: int

    @property
    def slug(self) -> str:
        return f"{slugify(self.cpu, max_length=40)}-{slugify(self.os, max_length=10)}"


def fingerprint(*, provider: str, intra_op_threads: int) -> Fingerprint:
    """Collect the host description. The only function in the project that reads the machine."""
    import onnxruntime as ort
    import psutil

    return Fingerprint(
        cpu=_cpu_model(),
        arch=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_gb=_total_ram_gb(),
        os=platform.system(),
        os_release=platform.release(),
        python=platform.python_version(),
        onnxruntime=ort.__version__,
        provider=provider,
        intra_op_threads=intra_op_threads,
    )


@dataclass
class PrecisionRow:
    """One measured (model, precision) combination."""

    precision: str
    size_mb: float
    latency_s_mean: float
    latency_s_std: float
    tokens_per_second: float
    peak_ram_mb: float
    perplexity: float | None
    generated_tokens: int
    measured_runs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultCard:
    """A full submission: one machine, one model, every precision it measured."""

    schema_version: int
    model_id: str
    revision: str
    created_utc: str
    tool_version: str
    machine: Fingerprint
    rows: list[PrecisionRow]
    eval_windows: int
    eval_corpus_sha256: str
    warmup_runs: int
    gen_tokens: int
    prompt_sha256: str
    notes: str = ""
    submitted_by: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["machine"] = asdict(self.machine)
        payload["rows"] = [r.as_dict() for r in self.rows]
        return payload

    @property
    def filename(self) -> str:
        """``<cpu>-<os>--<model>.json`` — stable, so a re-run updates its own card."""
        return f"{self.machine.slug}--{slugify(self.model_id.split('/')[-1], max_length=32)}.json"

    def speedup_vs(self, baseline: str = "fp32") -> dict[str, float]:
        """Throughput of each precision relative to the baseline precision."""
        by_precision = {r.precision: r for r in self.rows}
        base = by_precision.get(baseline)
        if base is None or base.tokens_per_second <= 0:
            return {}
        return {
            r.precision: round(r.tokens_per_second / base.tokens_per_second, 3)
            for r in self.rows
            if r.precision != baseline
        }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
