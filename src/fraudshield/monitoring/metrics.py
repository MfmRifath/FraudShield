"""Drift results as Prometheus metrics.

Prefect's ``serve`` executes each flow run in a subprocess, so gauges set inside the drift
flow would never reach the worker's /metrics. The flow writes ``<report_dir>/latest.json``
instead, and this collector (registered in the long-lived worker process) reads it on scrape.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from prometheus_client import REGISTRY, CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector


class DriftFileCollector(Collector):
    def __init__(self, latest_json: Path) -> None:
        self.path = latest_json

    def collect(self) -> Iterator[GaugeMetricFamily]:
        if not self.path.is_file():
            return
        try:
            d = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        detected = GaugeMetricFamily(
            "drift_detected", "1 if the last drift run triggered retraining", labels=["reason"]
        )
        reasons = [t.split(":", 1)[0] for t in d.get("triggers", [])]
        detected.add_metric(["any"], 1.0 if reasons else 0.0)
        for r in ("feature drift", "score drift", "performance"):
            detected.add_metric([r.replace(" ", "_")], 1.0 if r in reasons else 0.0)
        yield detected
        for name, value, doc in (
            (
                "drift_feature_share",
                d.get("drift_share"),
                "Share of top-k features drifted (Evidently)",
            ),
            (
                "drift_score_psi",
                d.get("score_psi"),
                "PSI of served probabilities, current vs reference",
            ),
            (
                "drift_pr_auc_relative_drop",
                (d.get("pr_auc") or {}).get("relative_drop"),
                "Relative matured-label PR-AUC drop",
            ),
        ):
            if value is not None:
                g = GaugeMetricFamily(name, doc)
                g.add_metric([], float(value))
                yield g
        as_of = d.get("as_of")
        if as_of:
            g = GaugeMetricFamily(
                "drift_window_end_timestamp_seconds", "Event-time end of the last drift window"
            )
            g.add_metric([], datetime.fromisoformat(as_of).timestamp())
            yield g
        g = GaugeMetricFamily(
            "drift_report_mtime_seconds", "When the last drift report was written"
        )
        g.add_metric([], self.path.stat().st_mtime)
        yield g


def serve_drift_metrics(
    report_dir: Path, port: int, registry: CollectorRegistry = REGISTRY
) -> None:
    registry.register(DriftFileCollector(report_dir / "latest.json"))
    start_http_server(port, registry=registry)
