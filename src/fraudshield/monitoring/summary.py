"""Live service summary for the UI's Model & Monitoring page (from the API's own metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry


def _samples(registry: CollectorRegistry, name: str) -> list[tuple[dict[str, str], float]]:
    return [
        (dict(s.labels), s.value) for m in registry.collect() for s in m.samples if s.name == name
    ]


def histogram_quantiles(
    registry: CollectorRegistry, metric: str, qs: tuple[float, ...] = (0.5, 0.95, 0.99)
) -> dict[str, float | None]:
    """Quantiles of a histogram aggregated over all label sets (linear interpolation, like PromQL)."""
    buckets: dict[float, float] = {}
    for labels, v in _samples(registry, f"{metric}_bucket"):
        le = float(labels["le"])
        buckets[le] = buckets.get(le, 0.0) + v
    edges = sorted(buckets)
    total = buckets.get(math.inf, 0.0)
    out: dict[str, float | None] = {}
    for q in qs:
        key = f"p{round(q * 100)}"
        if total == 0:
            out[key] = None
            continue
        rank, prev_le, prev_c = q * total, 0.0, 0.0
        for le in edges:
            c = buckets[le]
            if c >= rank:
                out[key] = (
                    prev_le
                    if math.isinf(le)
                    else prev_le + (le - prev_le) * (rank - prev_c) / max(c - prev_c, 1e-12)
                )
                break
            prev_le, prev_c = le, c
    return out


def counter_by(registry: CollectorRegistry, metric: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for labels, v in _samples(registry, f"{metric}_total"):
        out[labels.get(label, "")] = out.get(labels.get(label, ""), 0.0) + v
    return out


def latest_drift(report_dir: Path) -> dict[str, Any] | None:
    path = report_dir / "latest.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    keep = ("as_of", "reference_window", "current_window", "rows", "drift_share", "score_psi", "decision_mix",
            "pr_auc", "triggers", "skipped", "should_retrain")  # fmt: skip
    out = {k: d.get(k) for k in keep}
    html = d.get("report_html")
    out["has_html"] = bool(html) and (report_dir / Path(str(html)).name).is_file()
    out["drifted_columns"] = [c for c in d.get("drifted_columns", []) if c.get("drifted")]
    return out
