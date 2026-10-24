"""Backend for the analyst UI: cases -> labels -> training set (Phase 7 DoD), context, monitoring, policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Histogram

from fraudshield.mlops.training_set import ANALYST_SCHEMA
from fraudshield.monitoring.summary import histogram_quantiles
from fraudshield.serving.app import create_app
from fraudshield.synthetic import synthetic_canonical
from tests.case_contract import assert_reaches_training_set
from tests.serving_support import (
    ALWAYS_REVIEW_RULES,
    make_bundle,
    payload,
    serving_settings,
    write_yaml,
)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("ui")
    return make_bundle(serving_settings(tmp), tmp / "bundle")


@pytest.fixture(scope="module")
def canon() -> pl.DataFrame:
    return synthetic_canonical(10_000, seed=0).head(400)


def _client(tmp: Path, bundle: Path, **kw: Any) -> TestClient:
    rules = write_yaml(tmp / "rules.yaml", ALWAYS_REVIEW_RULES)
    return TestClient(create_app(serving_settings(tmp, bundle_dir=bundle, rules=rules, **kw)))


@pytest.mark.unit
def test_label_in_ui_appears_in_next_training_set(
    bundle_dir: Path, canon: pl.DataFrame, tmp_path: Path
) -> None:
    """Phase 7 DoD: an analyst label posted through the API reaches build_training_set."""
    with _client(tmp_path, bundle_dir) as c:
        for row in canon.to_dicts():
            c.post("/score", json=payload(row))
        queue = c.get("/cases", params={"limit": 5}).json()
        target = queue[0]["case_id"]
        assert (
            c.post(
                f"/cases/{target}/label", json={"label": "fraud", "analyst_id": "ana"}
            ).status_code
            == 200
        )
        store = c.app.state.fs.cases  # type: ignore[attr-defined]
        analyst = pl.DataFrame(c.portal.call(store.analyst_labels), schema=ANALYST_SCHEMA)  # type: ignore[union-attr]
    assert analyst["txn_id"].to_list() == [target]
    assert_reaches_training_set(canon, analyst, target)


@pytest.mark.unit
def test_case_context_timeline_linked_entities_and_audit(
    bundle_dir: Path, canon: pl.DataFrame, tmp_path: Path
) -> None:
    with _client(tmp_path, bundle_dir) as c:
        for row in canon.to_dicts():
            c.post("/score", json=payload(row))
        cases = c.get("/cases", params={"limit": 500}).json()
        # ring events are declined (no case); link through the email domain shared by many cards
        by_domain = canon.group_by("p_email_domain").agg(pl.col("uid").unique())
        sharers = {d: set(u) for d, u in by_domain.iter_rows()}
        case = max(cases, key=lambda x: len(sharers[x["txn"]["p_email_domain"]]))
        ctx = c.get(f"/cases/{case['case_id']}/context").json()
        c.post(
            f"/cases/{case['case_id']}/label",
            json={"label": "legit", "analyst_id": "bob", "notes": "known customer"},
        )
        audit = c.get(f"/cases/{case['case_id']}/audit").json()
        assert c.get("/cases/nope/context").status_code == 404
    assert ctx["uid"] == case["txn"]["uid"]
    assert ctx["store_available"] is True
    assert 1 <= len(ctx["timeline"]) <= 20
    assert any(t["txn_id"] == case["case_id"] for t in ctx["timeline"])
    times = [t["event_time"] for t in ctx["timeline"]]
    assert times == sorted(times, reverse=True)  # newest first
    others = sharers[case["txn"]["p_email_domain"]] - {case["txn"]["uid"]}
    assert ctx["linked"]["p_email_domain"]
    assert set(ctx["linked"]["p_email_domain"]) <= others
    assert [a["action"] for a in audit] == ["open", "label"]
    assert audit[1]["actor"] == "bob"


@pytest.mark.unit
def test_monitoring_summary_and_drift_report(
    bundle_dir: Path, canon: pl.DataFrame, tmp_path: Path
) -> None:
    report_dir = tmp_path / "drift"
    report_dir.mkdir()
    s_base = serving_settings(tmp_path, bundle_dir=bundle_dir)
    s = s_base.model_copy(
        update={"monitoring": s_base.monitoring.model_copy(update={"report_dir": report_dir})}
    )
    with TestClient(create_app(s)) as c:
        for row in canon.head(50).to_dicts():
            c.post("/score", json=payload(row))
        before = c.get("/monitoring/summary").json()
        assert c.get("/monitoring/drift/latest.html").status_code == 404
        (report_dir / "drift-x.html").write_text("<html>evidently</html>")
        (report_dir / "latest.json").write_text(json.dumps({"as_of": "2021-03-01T00:00:00+00:00", "drift_share": 0.4,
            "report_html": "/somewhere/else/drift-x.html", "triggers": ["feature drift: 40%"],
            "drifted_columns": [{"column": "amount", "drifted": True}, {"column": "hour", "drifted": False}]}))  # fmt: skip
        after = c.get("/monitoring/summary").json()
        html = c.get("/monitoring/drift/latest.html")
    assert sum(before["decisions"].values()) == 50
    assert before["models"]["champion"] == "synthetic-20x15"
    lat = before["latency_s"]
    assert 0 < lat["p50"] <= lat["p95"] <= lat["p99"]
    assert before["drift"] is None
    assert after["drift"]["has_html"] is True
    assert [c["column"] for c in after["drift"]["drifted_columns"]] == ["amount"]
    assert html.status_code == 200
    assert "evidently" in html.text  # served by basename from the report dir only


@pytest.mark.unit
def test_policy_simulator(bundle_dir: Path, tmp_path: Path) -> None:
    with TestClient(create_app(serving_settings(tmp_path, bundle_dir=bundle_dir))) as c:
        current = c.get("/policy/current").json()
        same = c.post(
            "/policy/simulate",
            json={"review_capacity_per_day": current["costs"]["review_capacity_per_day"]},
        ).json()
        none = c.post("/policy/simulate", json={"review_capacity_per_day": 0}).json()
        pricey = c.post("/policy/simulate", json={"analyst_cost": 1000}).json()
        bad = c.post("/policy/simulate", json={"margin_rate": -1})
    assert current["window"]["rows"] > 0
    assert same["simulated"]["outcome"] == same["current"]["outcome"] == current["outcome"]
    assert none["simulated"]["outcome"]["n_review"] == 0
    assert pricey["simulated"]["costs"]["analyst_cost"] == 1000
    assert pricey["simulated"]["outcome"]["n_review"] <= pricey["current"]["outcome"]["n_review"]
    assert bad.status_code == 422


@pytest.mark.unit
def test_histogram_quantiles_matches_bucket_interpolation() -> None:
    reg = CollectorRegistry()
    h = Histogram("lat_seconds", "x", ["v"], buckets=(0.01, 0.02, 0.05), registry=reg)
    for _ in range(50):
        h.labels(v="a").observe(0.005)  # bucket (0, 0.01]
    for _ in range(50):
        h.labels(v="b").observe(0.015)  # bucket (0.01, 0.02]
    q = histogram_quantiles(reg, "lat_seconds", (0.25, 0.5, 0.75))
    assert q == {
        "p25": pytest.approx(0.005),
        "p50": pytest.approx(0.01),
        "p75": pytest.approx(0.015),
    }
    assert histogram_quantiles(CollectorRegistry(), "lat_seconds") == {
        "p50": None,
        "p95": None,
        "p99": None,
    }
