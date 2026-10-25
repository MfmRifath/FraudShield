"""Drift detection, drift metrics, fraud-ring injection and monitoring config consistency."""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
import pytest
import yaml
from prometheus_client import CollectorRegistry, generate_latest

from fraudshield.config import MonitoringConfig
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.model_bundle import ModelBundle
from fraudshield.monitoring.drift import (
    decision_mix,
    detect_drift,
    matured_pr_auc,
    psi,
    top_features,
)
from fraudshield.monitoring.metrics import DriftFileCollector
from fraudshield.streaming.parquet_sink import HourlyParquetWriter
from fraudshield.streaming.replayer import inject_fraud_ring
from fraudshield.synthetic import EXTRAS, synthetic_canonical
from tests.serving_support import REPO, make_bundle, serving_settings

T0 = datetime(2021, 1, 1, tzinfo=UTC)


# ------------------------------------------------------------------ psi / mix / pr-auc
@pytest.mark.unit
def test_psi_properties() -> None:
    rng = np.random.default_rng(0)
    a = rng.beta(2, 20, 5000)
    assert psi(a, rng.beta(2, 20, 5000)) < 0.02
    assert psi(a, rng.beta(6, 10, 5000)) > 0.2
    assert np.isnan(psi([], a))
    assert psi(np.r_[a, np.nan], a) == pytest.approx(psi(a, a))


@pytest.mark.unit
def test_decision_mix_and_matured_pr_auc() -> None:
    d = pl.DataFrame({"txn_id": [f"t{i}" for i in range(6)], "decision": ["approve"] * 4 + ["review", "block"],
                      "probability": [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]})  # fmt: skip
    assert decision_mix(d) == {"approve": 4 / 6, "review": 1 / 6, "block": 1 / 6}
    labels = pl.DataFrame(
        {"txn_id": [f"t{i}" for i in range(6)], "label": [0, 0, 0, 1, 1, 0],
         "label_available_time": [T0] * 5 + [T0 + timedelta(days=9)]},
        schema={"txn_id": pl.String, "label": pl.Int8, "label_available_time": pl.Datetime("us", "UTC")},
    )  # fmt: skip
    pr, n, pos = matured_pr_auc(d, labels, T0 + timedelta(days=1), min_pos=2)
    assert (n, pos) == (5, 2)  # t5's label has not matured yet
    assert pr == pytest.approx(1.0)
    assert matured_pr_auc(d, labels, T0 + timedelta(days=1), min_pos=3)[0] is None


# ------------------------------------------------------------------ drift end to end (synthetic)
def _decision_log(canon: pl.DataFrame, bundle: ModelBundle) -> pl.DataFrame:
    """What the parquet sink would have logged if ``canon`` had been served by ``bundle``."""
    fs = build_feature_set(EXTRAS, 50.0)
    feats = compute_offline(canon, fs)
    p = bundle.predict_proba(feats)
    return pl.DataFrame({
        "txn_id": feats["txn_id"], "uid": feats["uid"], "event_time": feats["event_time"], "amount": feats["amount"],
        "decision": np.where(p > 0.5, "block", "approve"), "probability": p,
        "features": [orjson.dumps(r, default=str).decode() for r in feats.select(fs.names).to_dicts()],
    })  # fmt: skip


@pytest.fixture(scope="module")
def drift_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("drift")
    bundle = ModelBundle.load(make_bundle(serving_settings(tmp), tmp / "bundle", rounds=30))
    canon = synthetic_canonical(10_000, seed=0)
    # ring only in the last 15 days = the current window of WINDOWS; reference stays clean
    split = canon["event_time"].max() - timedelta(days=15)  # type: ignore[operator]
    t = pl.col("event_time")
    ref, cur = canon.filter(t < split), canon.filter(t >= split)
    drifted = pl.concat(
        [ref, pl.DataFrame([inject_fraud_ring(r, 0.4) for r in cur.to_dicts()], schema=cur.schema)]
    )
    return {
        "bundle": bundle,
        "clean": _decision_log(canon, bundle),
        "drifted": _decision_log(drifted, bundle),
        "tmp": tmp,
    }


def _labels(log: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "txn_id": pl.String,
            "label": pl.Int8,
            "label_available_time": pl.Datetime("us", "UTC"),
        }
    )


@pytest.mark.unit
def test_no_drift_on_stationary_traffic(drift_setup: dict[str, Any]) -> None:
    # Both windows after the history warm-up: early in a log, first-seen flags, 7-day windows and
    # target encodings are still filling up, which IS a distribution change (an earlier version of
    # this test used a reference window from day 10 and correctly saw 36 % of features drift).
    cfg = MonitoringConfig(reference_days=15, current_days=15)
    res = detect_drift(
        drift_setup["clean"], _labels(drift_setup["clean"]), drift_setup["bundle"], cfg
    )
    assert res.drift_share is not None
    assert res.drift_share <= cfg.drift_share_threshold, res.drifted_columns
    assert res.score_psi is not None
    assert res.score_psi < cfg.psi_threshold
    assert not res.should_retrain
    assert "PR-AUC check" in res.skipped[0]


@pytest.mark.unit
def test_fraud_ring_triggers_retrain_and_writes_reports(drift_setup: dict[str, Any]) -> None:
    cfg = MonitoringConfig(reference_days=15, current_days=15)  # same windows as the no-drift test
    out = drift_setup["tmp"] / "reports"
    res = detect_drift(
        drift_setup["drifted"],
        _labels(drift_setup["drifted"]),
        drift_setup["bundle"],
        cfg,
        out_dir=out,
    )
    assert res.should_retrain
    assert any(t.startswith("feature drift") for t in res.triggers), res.as_dict()
    assert len(res.features) == min(20, len(top_features(drift_setup["bundle"], 99)))
    assert res.report_html is not None
    assert Path(res.report_html).stat().st_size > 10_000  # Evidently HTML report
    latest = json.loads((out / "latest.json").read_text())
    assert latest["should_retrain"] is True
    assert latest["triggers"] == res.triggers


@pytest.mark.unit
def test_small_windows_are_skipped(drift_setup: dict[str, Any]) -> None:
    cfg = MonitoringConfig(reference_days=0.01, current_days=0.01, min_rows=200)
    res = detect_drift(
        drift_setup["clean"], _labels(drift_setup["clean"]), drift_setup["bundle"], cfg
    )
    assert res.drift_share is None
    assert not res.should_retrain
    assert "fewer than 200 rows" in res.skipped[0]


# ------------------------------------------------------------------ metrics collector
@pytest.mark.unit
def test_drift_file_collector(tmp_path: Path) -> None:
    reg = CollectorRegistry()
    reg.register(DriftFileCollector(tmp_path / "latest.json"))
    assert "drift_detected" not in generate_latest(reg).decode()  # no report yet -> no series
    (tmp_path / "latest.json").write_text(json.dumps({
        "as_of": "2021-03-01T00:00:00+00:00", "drift_share": 0.55, "score_psi": 0.31,
        "pr_auc": {"relative_drop": None}, "triggers": ["feature drift: 55%", "score drift: PSI 0.31"],
    }))  # fmt: skip
    text = generate_latest(reg).decode()
    assert 'drift_detected{reason="any"} 1.0' in text
    assert 'drift_detected{reason="feature_drift"} 1.0' in text
    assert 'drift_detected{reason="performance"} 0.0' in text
    assert "drift_feature_share 0.55" in text
    assert "drift_pr_auc_relative_drop" not in text


# ------------------------------------------------------------------ injection + sink
@pytest.mark.unit
def test_fraud_ring_injection_is_deterministic() -> None:
    rows = synthetic_canonical(2_000, seed=1).to_dicts()
    a = [inject_fraud_ring(r, 0.3) for r in rows]
    assert a == [inject_fraud_ring(r, 0.3) for r in rows]
    ring = [r for r in a if r["p_email_domain"] == "ring-mail.example"]
    assert 0.25 < len(ring) / len(rows) < 0.35
    assert {r["device_info"] for r in ring} <= {"ring-device-0", "ring-device-1", "ring-device-2"}
    assert all(r["label"] == 1 for r in ring)


@pytest.mark.unit
def test_parquet_sink_logs_labels(tmp_path: Path) -> None:
    w = HourlyParquetWriter(tmp_path, flush_s=60, max_rows=1, kind="labels")
    w.add(
        {
            "txn_id": "t",
            "uid": "u",
            "label": 1,
            "label_source": "chargeback",
            "label_available_time": T0.isoformat(),
        }
    )
    df = pl.read_parquet(tmp_path / "labels" / "**" / "*.parquet")
    assert df.to_dicts() == [
        {
            "txn_id": "t",
            "uid": "u",
            "label": 1,
            "label_source": "chargeback",
            "label_available_time": T0,
        }
    ]


# ------------------------------------------------------------------ monitoring config consistency
def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_dashboards", REPO / "scripts/build_dashboards.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exported_metrics() -> set[str]:
    from fraudshield.serving.metrics import ServingMetrics
    from fraudshield.streaming.common import WorkerMetrics

    names = {m.name for m in ServingMetrics().registry.collect()} | {
        m.name for m in WorkerMetrics("w").registry.collect()
    }
    names |= {"shadow_comparisons", "shadow_abs_prob_diff", "shadow_challenger_info", "drift_detected", "drift_feature_share",
              "drift_score_psi", "drift_pr_auc_relative_drop", "up"}  # fmt: skip
    return names


_PROMQL_WORDS = {"sum", "max", "min", "rate", "by", "le", "histogram_quantile", "clamp_min", "ignoring", "group_left",
                 "without", "on", "avg", "count"}  # fmt: skip


def _metric_names(expr: str) -> set[str]:
    expr = re.sub(r'\{[^}]*\}|"[^"]*"|\[[^\]]*\]', "", expr)
    expr = re.sub(r"\b\d+(\.\d+)?(e[-+]?\d+)?\b", "", expr)  # numeric literals such as 1e-9
    expr = re.sub(
        r"\b(by|without|ignoring|on|group_left|group_right)\s*\([^)]*\)", "", expr
    )  # label lists
    return {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr) if t not in _PROMQL_WORDS}


def _base(name: str) -> str:
    return re.sub(r"_(bucket|count|sum|total|created)$", "", name)


@pytest.mark.unit
def test_dashboards_are_generated_and_use_exported_metrics() -> None:
    built = _load_builder().build()
    on_disk = {p.name: p.read_text() for p in (REPO / "infra/grafana/dashboards").glob("*.json")}
    assert built == on_disk, "dashboards out of date: run `make dashboards`"
    exported = {_base(n) for n in _exported_metrics()}
    uids = set()
    for text in built.values():
        dash = json.loads(text)
        uids.add(dash["uid"])
        for panel in dash["panels"]:
            for target in panel["targets"]:
                unknown = {n for n in _metric_names(target["expr"]) if _base(n) not in exported}
                assert not unknown, (dash["title"], panel["title"], unknown)
    assert len(uids) == 4


@pytest.mark.unit
def test_alert_rules_match_the_spec() -> None:
    rules = {
        r["alert"]: r
        for g in yaml.safe_load((REPO / "infra/prometheus/rules/fraudshield.yml").read_text())[
            "groups"
        ]
        for r in g["rules"]
    }
    assert "> 0.1" in rules["ScoreLatencyP95High"]["expr"]
    assert rules["ScoreLatencyP95High"]["for"] == "5m"
    assert "> 0.01" in rules["FallbackRateHigh"]["expr"]
    assert "> 30" in rules["ConsumerLagHigh"]["expr"]
    assert "drift_detected" in rules["DriftDetected"]["expr"]
    exported = {_base(n) for n in _exported_metrics()}
    for r in rules.values():
        assert {_base(n) for n in _metric_names(r["expr"])} <= exported, r["alert"]


@pytest.mark.unit
def test_prometheus_scrapes_every_compose_service_with_metrics() -> None:
    cfg = yaml.safe_load((REPO / "infra/prometheus/prometheus.yml").read_text())
    targets = {
        t for job in cfg["scrape_configs"] for sc in job["static_configs"] for t in sc["targets"]
    }
    compose = yaml.safe_load((REPO / "compose.yml").read_text())["services"]
    for svc in (
        "processor",
        "gateway",
        "label-emitter",
        "parquet-sink",
        "shadow",
        "prefect-worker",
    ):
        assert svc in compose
        assert f"{svc}:9100" in targets
    assert "api:8000" in targets


@pytest.mark.unit
def test_daily_job_measures_performance_on_matured_labels(drift_setup: dict[str, Any]) -> None:
    """Regression: with a 1-day current window no label had matured (delays are 1-7 days), so
    the PR-AUC check was always skipped; it now uses windows lagged by the label delay."""
    canon = synthetic_canonical(10_000, seed=0)
    labels = canon.filter(pl.col("label").is_not_null()).select(
        "txn_id", label=pl.col("label").cast(pl.Int8), label_available_time="label_available_time"
    )
    daily = MonitoringConfig(reference_days=7, current_days=1, min_rows=50, min_label_positives=5)
    fresh = detect_drift(drift_setup["clean"], labels, drift_setup["bundle"], daily,
                         performance_days=1, performance_lag_days=0)  # fmt: skip
    assert fresh.pr_auc["current"] is None  # today's labels have not arrived yet
    lagged = detect_drift(drift_setup["clean"], labels, drift_setup["bundle"], daily)
    assert lagged.pr_auc["current"] is not None
    assert lagged.pr_auc["reference"] is not None
    assert not any("PR-AUC" in s for s in lagged.skipped)
    end = drift_setup["clean"]["event_time"].max()
    assert lagged.pr_auc["windows"]["current"][1] == (end - timedelta(days=7)).isoformat()
