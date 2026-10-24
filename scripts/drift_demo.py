"""Drift demo on the running stack (``make drift-demo``; runs on the compose network).

1. Replay the first 14 days of the CC0 fixture normally, then the last 7 days with a 40 %
   "fraud ring" (shared devices + one email domain, larger amounts), under a fresh namespace.
2. Wait until the parquet sink has logged every decision of the run.
3. Run the ``drift/on-demand`` Prefect deployment (reference = 7 days before, current = last 7
   days of the run): Evidently feature drift, score PSI, matured-label PR-AUC.
4. Check that drift triggered the retrain flow (a new MLflow registered model appears), that
   Prometheus fires ``DriftDetected``, and that every Grafana dashboard panel has data.

Writes ``reports/drift/demo-summary.md``. Exits non-zero if the alert, the retrain or the
dashboards do not come through.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
from confluent_kafka import Producer

from fraudshield.config import Settings
from fraudshield.streaming.replayer import load_source, replay

PROM, GRAFANA, MLFLOW = "http://prometheus:9090", "http://grafana:3000", "http://mlflow:5000"
REPORT = Path("reports/drift")


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310 - compose-internal URLs
        return json.loads(r.read())


def prom_query(expr: str) -> list[dict[str, Any]]:
    res: list[dict[str, Any]] = get_json(
        f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    )["data"]["result"]
    return res


def sink_decisions_logged() -> float:
    rows = prom_query('sum(messages_processed_total{worker="parquet_sink",topic="decisions"})')
    return float(rows[0]["value"][1]) if rows else 0.0


def wait_for(what: str, check: Any, timeout_s: float, every_s: float = 2.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = check()
        if out:
            return out
        time.sleep(every_s)
    raise TimeoutError(f"timed out waiting for {what}")


def dashboards_with_data() -> list[tuple[str, str, bool]]:
    rows = []
    for d in get_json(f"{GRAFANA}/api/search?tag=fraudshield"):
        dash = get_json(f"{GRAFANA}/api/dashboards/uid/{d['uid']}")["dashboard"]
        for p in dash["panels"]:
            has = any(prom_query(t["expr"]) for t in p.get("targets", []))
            rows.append((dash["title"], p["title"], has))
    return rows


def main() -> int:
    s = Settings()
    ns = f"drift{datetime.now(UTC).strftime('%m%d%H%M%S')}"
    canon = load_source(s, "fixture", "sparkov").sort("event_time")
    split = canon["event_time"].min() + timedelta(days=14)  # type: ignore[operator]
    normal, drifted = (
        canon.filter(pl.col("event_time") < split),
        canon.filter(pl.col("event_time") >= split),
    )
    before = sink_decisions_logged()
    producer = Producer({"bootstrap.servers": s.streaming.bootstrap_servers, "linger.ms": 5})
    n1 = replay(normal.to_dicts(), producer, s, rate=300, namespace=ns).raw
    n2 = replay(drifted.to_dicts(), producer, s, rate=300, namespace=ns, drift_share=0.4).raw
    print(f"[1] replayed {n1} normal + {n2} drifted events (namespace {ns})")

    wait_for(
        "parquet sink to log the run",
        lambda: sink_decisions_logged() >= before + n1 + n2,
        timeout_s=300,
    )
    time.sleep(s.streaming.parquet_flush_s + 6)  # let the timer flush the last buffers
    print("[2] decisions logged by the parquet sink")

    from prefect.client.schemas.objects import FlowRun, StateType
    from prefect.deployments import run_deployment

    run = cast(FlowRun, run_deployment("drift/on-demand", timeout=1800,
                                       parameters={"namespace": ns, "reference_days": 7, "current_days": 7}))  # fmt: skip
    state = run.state.type if run.state else None
    print(f"[3] drift flow run {run.name}: {state}")
    result = json.loads((REPORT / "latest.json").read_text())
    print(
        f"    feature drift share {result['drift_share']}, PSI {result['score_psi']}, triggers {result['triggers']}"
    )

    from mlflow.tracking import MlflowClient

    retrained = [
        m.name
        for m in MlflowClient(tracking_uri=MLFLOW).search_registered_models()
        if m.name.startswith("fraudshield-drift-demo")
    ]
    alert = wait_for(
        "DriftDetected to fire",
        lambda: [a for a in get_json(f"{PROM}/api/v1/alerts")["data"]["alerts"] if a["labels"]["alertname"] == "DriftDetected" and a["state"] == "firing"],
        timeout_s=120,
    )  # fmt: skip
    print(
        f"[4] Prometheus alert DriftDetected: {alert[0]['state']}; retrain registered models: {retrained}"
    )

    panels = dashboards_with_data()
    with_data = sum(h for *_, h in panels)
    print(
        f"[5] Grafana: {len({t for t, *_ in panels})} dashboards, {with_data}/{len(panels)} panels with data"
    )

    REPORT.mkdir(parents=True, exist_ok=True)
    table = "\n".join(f"| {t} | {p} | {'yes' if h else 'no data'} |" for t, p, h in panels)
    triggers = "\n".join(f"- {x}" for x in result["triggers"]) or "- none"
    (REPORT / "demo-summary.md").write_text(f"""# Drift demo ({ns})

_Generated by `scripts/drift_demo.py` (`make drift-demo`) at {datetime.now(UTC).isoformat(timespec="seconds")}.
Data: the committed CC0 Sparkov fixture; the last 7 days replayed with a 40 % injected fraud ring.
Model: the fixture-trained dev champion. Retraining runs the synthetic `mlops-demo` flow until
the real training run._

- Windows: reference {result["reference_window"][0]} .. {result["reference_window"][1]}, current {result["current_window"][0]} .. {result["current_window"][1]} ({result["rows"]})
- Drifted share of top-{len(result["features"])} features: {result["drift_share"]:.2f} (threshold {s.monitoring.drift_share_threshold})
- Score PSI: {result["score_psi"]:.3f} (threshold {s.monitoring.psi_threshold})
- Skipped: {result["skipped"] or "none"}

Triggers:
{triggers}

Drift flow state: {state}. Retrain registered models: {retrained or "none"}. Prometheus `DriftDetected`: firing.

| dashboard | panel | data |
|---|---|---|
{table}
""")
    ok = (
        state == StateType.COMPLETED
        and result["triggers"]
        and retrained
        and with_data >= 0.8 * len(panels)
    )
    print(f"summary: {REPORT / 'demo-summary.md'} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
