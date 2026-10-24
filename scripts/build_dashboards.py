"""Generate the Grafana dashboards in infra/grafana/dashboards/ (run: make dashboards).

Dashboards are code: panels are declared here and rendered to JSON, and
tests/unit/test_monitoring_config.py checks the committed JSON is up to date and that every
query only uses metrics the services export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OUT = Path("infra/grafana/dashboards")
DS = {"type": "prometheus", "uid": "prometheus"}


def ts(
    title: str, *exprs: tuple[str, str], unit: str = "short", w: int = 12, h: int = 8
) -> dict[str, Any]:
    return {
        "type": "timeseries",
        "title": title,
        "w": w,
        "h": h,
        "unit": unit,
        "exprs": list(exprs),
    }


def stat(title: str, expr: str, unit: str = "short", w: int = 6, h: int = 4) -> dict[str, Any]:
    return {"type": "stat", "title": title, "w": w, "h": h, "unit": unit, "exprs": [(expr, "")]}


def q(expr: str, legend: str = "") -> tuple[str, str]:
    return (expr, legend)


P95 = "histogram_quantile({q}, sum by (le) (rate(score_latency_seconds_bucket[1m])))"
FALLBACK_RATE = 'sum(rate(fallback_total{reason!="review_capacity"}[5m])) / clamp_min(sum(rate(decisions_total[5m])), 1e-9)'

DASHBOARDS: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "service-slos": (
        "FraudShield / Service SLOs",
        [
            stat("p95 /score latency (1m)", P95.format(q=0.95), unit="s"),
            stat("Requests / s", "sum(rate(decisions_total[1m]))", unit="reqps"),
            stat("Fallback rate (5m)", FALLBACK_RATE, unit="percentunit"),
            stat("API up", 'up{job="api"}'),
            ts(
                "/score latency quantiles",
                q(P95.format(q=0.5), "p50"),
                q(P95.format(q=0.95), "p95"),
                q(P95.format(q=0.99), "p99"),
                unit="s",
            ),
            ts(
                "Requests / s by degraded",
                q(
                    "sum by (degraded) (rate(score_latency_seconds_count[1m]))",
                    "degraded={{degraded}}",
                ),
                unit="reqps",
            ),
            ts(
                "Fallbacks / s by reason",
                q("sum by (reason) (rate(fallback_total[1m]))", "{{reason}}"),
                unit="reqps",
            ),
            ts(
                "Feature fetch p95",
                q(
                    "histogram_quantile(0.95, sum by (le) (rate(feature_fetch_seconds_bucket[1m])))",
                    "p95",
                ),
                unit="s",
            ),
        ],
    ),
    "decisions": (
        "FraudShield / Decision & Score mix",
        [
            ts(
                "Decisions / s",
                q("sum by (decision) (rate(decisions_total[1m]))", "{{decision}}"),
                unit="reqps",
            ),
            ts(
                "Decision share (5m)",
                q(
                    "sum by (decision) (rate(decisions_total[5m])) / ignoring(decision) group_left sum(rate(decisions_total[5m]))",
                    "{{decision}}",
                ),
                unit="percentunit",
            ),
            ts(
                "Fraud score quantiles",
                q("histogram_quantile(0.5, sum by (le) (rate(fraud_score_bucket[5m])))", "p50"),
                q("histogram_quantile(0.9, sum by (le) (rate(fraud_score_bucket[5m])))", "p90"),
                q("histogram_quantile(0.99, sum by (le) (rate(fraud_score_bucket[5m])))", "p99"),
                unit="percentunit",
            ),
            ts("Review queue depth", q("review_queue_depth", "open cases")),
            ts(
                "Published events dropped / s",
                q("sum by (topic) (rate(publish_dropped_total[1m]))", "{{topic}}"),
            ),
        ],
    ),
    "champion-challenger": (
        "FraudShield / Champion vs Challenger",
        [
            stat("Champion version", 'max by (version) (model_version_info{alias="champion"})'),
            stat("Challenger (shadow)", "max by (version) (shadow_challenger_info)"),
            stat(
                "Decision agreement (15m)",
                'sum(rate(shadow_comparisons_total{agree="true"}[15m])) / clamp_min(sum(rate(shadow_comparisons_total[15m])), 1e-9)',
                unit="percentunit",
            ),
            ts(
                "Shadow comparisons / s",
                q("sum by (agree) (rate(shadow_comparisons_total[1m]))", "agree={{agree}}"),
            ),
            ts(
                "|p challenger - p champion|",
                q(
                    "histogram_quantile(0.5, sum by (le) (rate(shadow_abs_prob_diff_bucket[5m])))",
                    "p50",
                ),
                q(
                    "histogram_quantile(0.95, sum by (le) (rate(shadow_abs_prob_diff_bucket[5m])))",
                    "p95",
                ),
            ),
            ts(
                "Live traffic by model version",
                q(
                    "sum by (model_version) (rate(score_latency_seconds_count[1m]))",
                    "{{model_version}}",
                ),
                unit="reqps",
            ),
        ],
    ),
    "pipeline-health": (
        "FraudShield / Pipeline health",
        [
            stat("Drift detected", 'max(drift_detected{reason="any"})'),
            stat("Drifted feature share", "max(drift_feature_share)", unit="percentunit"),
            stat("Score PSI", "max(drift_score_psi)"),
            stat("Max consumer lag", "max(consumer_lag_seconds)", unit="s"),
            ts(
                "Consumer lag by worker/topic",
                q("max by (worker, topic) (consumer_lag_seconds)", "{{worker}} / {{topic}}"),
                unit="s",
            ),
            ts(
                "Messages / s by worker",
                q(
                    "sum by (worker, outcome) (rate(messages_processed_total[1m]))",
                    "{{worker}} {{outcome}}",
                ),
            ),
            ts("Scrape targets up", q("up", "{{job}} {{instance}}")),
        ],
    ),
}


def render(uid: str, title: str, panels: list[dict[str, Any]]) -> dict[str, Any]:
    out, x, y, row_h = [], 0, 0, 0
    for i, p in enumerate(panels, start=1):
        if x + p["w"] > 24:
            x, y, row_h = 0, y + row_h, 0
        out.append({
            "id": i, "type": p["type"], "title": p["title"], "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": p["w"], "h": p["h"]},
            "fieldConfig": {"defaults": {"unit": p["unit"]}, "overrides": []},
            "options": {} if p["type"] == "timeseries" else {"reduceOptions": {"calcs": ["lastNotNull"]}},
            "targets": [{"refId": chr(65 + j), "datasource": DS, "expr": e, "legendFormat": lg} for j, (e, lg) in enumerate(p["exprs"])],
        })  # fmt: skip
        x, row_h = x + p["w"], max(row_h, p["h"])
    return {"uid": f"fraudshield-{uid}", "title": title, "tags": ["fraudshield"], "schemaVersion": 41,
            "time": {"from": "now-30m", "to": "now"}, "refresh": "10s", "timezone": "utc", "editable": False,
            "panels": out}  # fmt: skip


def build() -> dict[str, str]:
    return {
        f"{uid}.json": json.dumps(render(uid, t, p), indent=2) + "\n"
        for uid, (t, p) in DASHBOARDS.items()
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in build().items():
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
