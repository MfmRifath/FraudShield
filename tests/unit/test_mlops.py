"""MLOps loop: training-set PIT labels, registry history, promotion gate, canary, demo, flow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import pytest
from fastapi.testclient import TestClient

from fraudshield.features.definitions import build_feature_set
from fraudshield.mlops.gate import run_gate
from fraudshield.mlops.registry import Registry, RegistryError
from fraudshield.mlops.training_set import (
    ANALYST_SCHEMA,
    analyst_labels_from_cases,
    build_training_set,
)
from fraudshield.model_bundle import ModelBundle
from fraudshield.pipeline import feature_set_for
from fraudshield.serving.app import create_app
from fraudshield.synthetic import (
    EXTRAS,
    build_synthetic_bundle,
    synthetic_canonical,
    synthetic_dataset_config,
    synthetic_features,
)
from fraudshield.training.costs import CostConfig
from fraudshield.training.package import log_and_register
from fraudshield.training.train import prepare_splits
from tests.serving_support import REPO, TOKEN, make_bundle, payload, serving_settings, window_rows

T0 = datetime(2021, 1, 1, tzinfo=UTC)


# ------------------------------------------------------------------ training set
def _feats(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"txn_id": pl.String, "event_time": pl.Datetime("us", "UTC"), "label": pl.Int8,
                "label_available_time": pl.Datetime("us", "UTC"), "amount": pl.Float64},
    )  # fmt: skip


@pytest.mark.unit
def test_training_set_pit_union_and_conflicts() -> None:
    d = timedelta(days=1)
    feats = _feats([
        {"txn_id": "a", "event_time": T0, "label": 1, "label_available_time": T0 + 2 * d, "amount": 1.0},      # matured cb
        {"txn_id": "b", "event_time": T0, "label": 0, "label_available_time": T0 + 9 * d, "amount": 1.0},      # immature cb, analyst says 1
        {"txn_id": "c", "event_time": T0, "label": 0, "label_available_time": T0 + 2 * d, "amount": 1.0},      # conflict: cb 0 wins
        {"txn_id": "d", "event_time": T0, "label": 1, "label_available_time": T0 + 9 * d, "amount": 1.0},      # immature, no analyst -> dropped
        {"txn_id": "e", "event_time": T0 + 6 * d, "label": 1, "label_available_time": T0 + 7 * d, "amount": 1.0},  # after as_of
    ])  # fmt: skip
    analyst = pl.DataFrame(
        [{"txn_id": "b", "label": 0, "label_time": T0 + 1 * d, "analyst_id": "x"},
         {"txn_id": "b", "label": 1, "label_time": T0 + 3 * d, "analyst_id": "y"},   # latest as of cutoff wins
         {"txn_id": "c", "label": 1, "label_time": T0 + 1 * d, "analyst_id": "x"},
         {"txn_id": "d", "label": 1, "label_time": T0 + 6 * d, "analyst_id": "x"}],  # labelled after as_of
        schema=ANALYST_SCHEMA,
    )  # fmt: skip
    out, stats = build_training_set(feats, analyst, as_of=T0 + 5 * d)
    got = {r["txn_id"]: (r["label"], r["label_source"]) for r in out.to_dicts()}
    assert got == {"a": (1, "chargeback"), "b": (1, "analyst"), "c": (0, "chargeback")}
    assert stats.conflicts == 1
    assert stats.chargeback_labels == 2
    assert stats.rows == 3
    assert out.columns == [*feats.columns, "label_source"]
    assert out.filter(pl.col("txn_id") == "b")["label_available_time"][0] == T0 + 3 * d


@pytest.mark.unit
def test_analyst_labels_from_cases() -> None:
    from fraudshield.cases.store import Case

    base: dict[str, Any] = {"created_at": T0, "txn": {}, "probability": 0.5, "decision": "review", "reasons": [],
            "expected_loss": 1.0, "model_version": "v"}  # fmt: skip
    cases = [
        Case(case_id="x", status="labeled", label="fraud", labeled_at=T0, analyst_id="a", **base),
        Case(case_id="y", **base),  # still open
    ]
    df = analyst_labels_from_cases(cases)
    assert df.to_dicts() == [{"txn_id": "x", "label": 1, "label_time": T0, "analyst_id": "a"}]


# ------------------------------------------------------------------ registry + gate
@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> tuple[Registry, dict[str, Path], Any]:
    tmp = tmp_path_factory.mktemp("reg")
    s = serving_settings(tmp)
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("t", artifact_location=(tmp / "art").resolve().as_uri())
    mlflow.set_experiment("t")
    # bundle dirs deliberately NOT named "bundle": MLflow stores artifacts under the dir name
    bundles = {
        "weak": make_bundle(s, tmp / "weak", rounds=2),
        "strong": make_bundle(s, tmp / "strong", rounds=40),
    }
    example = prepare_splits(synthetic_features(10_000, seed=0), synthetic_dataset_config())[
        "test"
    ].head(5)
    versions = {}
    for name in ("weak", "strong"):
        with mlflow.start_run():
            _, v = log_and_register(bundles[name], example, "gate-test", alias="challenger")
        versions[name] = v
    return Registry(uri, "gate-test"), bundles, versions


@pytest.mark.unit
def test_registry_promote_and_rollback_history(
    registry: tuple[Registry, dict[str, Path], Any],
) -> None:
    reg, _, v = registry
    assert reg.version_of("challenger") == v["strong"]
    assert reg.version_of("champion") is None
    assert reg.promote(v["weak"]).before is None
    with pytest.raises(RegistryError, match="already"):
        reg.promote(v["weak"])
    move = reg.promote(v["strong"])
    assert (move.before, move.after) == (v["weak"], v["strong"])
    assert reg.history() == [v["weak"]]
    back = reg.rollback()
    assert (back.before, back.after) == (v["strong"], v["weak"])
    assert reg.version_of("champion") == v["weak"]
    assert reg.history() == []
    with pytest.raises(RegistryError, match="no previous champion"):
        reg.rollback()
    assert isinstance(reg.load_bundle(v["strong"]), ModelBundle)


@pytest.fixture(scope="module")
def gate_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("gate")
    s = serving_settings(tmp)
    feats = synthetic_features(10_000, seed=0)
    return {
        "settings": s,
        # 45 % label noise: a genuinely worse model on the same data
        "weak": ModelBundle.load(
            build_synthetic_bundle(s, tmp / "weak", rounds=40, label_noise=0.45)
        ),
        "strong": ModelBundle.load(make_bundle(s, tmp / "strong", rounds=40)),
        "test": prepare_splits(feats, synthetic_dataset_config())["test"],
        "canon": synthetic_canonical(600, seed=0),
        "fs": feature_set_for(s, "sparkov"),
        "costs": CostConfig.load(REPO / "configs/costs.yaml"),
    }


def _gate(g: dict[str, Any], challenger: str, champion: str | None, **cfg: Any) -> Any:
    mcfg = g["settings"].mlops.model_copy(
        update={"parity_rows": 600, "latency_bench_rows": 100} | cfg
    )
    champ = g[champion] if champion else None
    return run_gate(g[challenger], champ, g["test"], g["canon"], g["fs"], g["costs"], mcfg)


@pytest.mark.unit
def test_gate_passes_for_better_challenger(gate_inputs: dict[str, Any]) -> None:
    res = _gate(gate_inputs, "strong", "weak")
    assert res.passed, res.to_markdown()
    assert {c.name for c in res.checks} == {"pr_auc", "net_savings", "latency", "parity"}
    assert 0.0 <= res.shadow["decision_agreement"] <= 1.0
    assert "Gate: PASSED" in res.to_markdown()
    # the gate times predict only (spec); explanations are measured and reported, not gated
    lat = next(c for c in res.checks if c.name == "latency")
    assert lat.value is not None
    assert (
        res.info["p95_with_3_reason_codes_ms"] > 0
    )  # (no timing comparison: noisy on tiny models)
    assert "p95_with_3_reason_codes_ms" in res.to_markdown()


@pytest.mark.unit
def test_gate_blocks_worse_challenger(gate_inputs: dict[str, Any]) -> None:
    res = _gate(gate_inputs, "weak", "strong")
    failed = {c.name for c in res.checks if not c.passed}
    assert "pr_auc" in failed
    assert not res.passed


@pytest.mark.unit
def test_gate_latency_and_skew_checks(gate_inputs: dict[str, Any]) -> None:
    slow = _gate(gate_inputs, "strong", None, max_single_row_p95_ms=1e-6)
    assert {c.name for c in slow.checks if not c.passed} == {"latency"}
    strong: ModelBundle = gate_inputs["strong"]
    meta = strong.spec.metadata | {"feature_set_hash": "deadbeef"}  # trained on another feature set
    skewed = ModelBundle(strong.spec.model_copy(update={"metadata": meta}), strong.boosters)
    res = run_gate(skewed, None, gate_inputs["test"], gate_inputs["canon"], gate_inputs["fs"], gate_inputs["costs"],
                   gate_inputs["settings"].mlops.model_copy(update={"latency_bench_rows": 50}))  # fmt: skip
    parity = next(c for c in res.checks if c.name == "parity")
    assert not parity.passed
    assert "deadbeef" in parity.detail
    assert build_feature_set(EXTRAS, 50.0).spec_hash() == gate_inputs["fs"].spec_hash()


# ------------------------------------------------------------------ canary
@pytest.mark.unit
def test_runtime_canary_toggle(tmp_path: Path) -> None:
    s0 = serving_settings(tmp_path)
    champ, chall = make_bundle(s0, tmp_path / "c1"), make_bundle(s0, tmp_path / "c2", rounds=10)
    rows = window_rows()[:200]
    hdr = {"X-Admin-Token": TOKEN}
    with TestClient(
        create_app(serving_settings(tmp_path, bundle_dir=champ, challenger_bundle_dir=chall))
    ) as c:
        assert {c.post("/score", json=payload(r)).json()["variant"] for r in rows[:50]} == {
            "champion"
        }
        assert c.post("/admin/canary", json={"pct": 100}).status_code == 401
        assert c.post("/admin/canary", json={"pct": 101}, headers=hdr).status_code == 422
        r = c.post("/admin/canary", json={"pct": 100}, headers=hdr)
        assert r.status_code == 200
        assert r.json()["canary_pct"] == 100
        assert {c.post("/score", json=payload(x)).json()["variant"] for x in rows[50:100]} == {
            "challenger"
        }
        assert c.post("/admin/canary", json={"pct": 0}, headers=hdr).json()["canary_pct"] == 0
    with TestClient(create_app(serving_settings(tmp_path, bundle_dir=champ))) as c:
        assert (
            c.post("/admin/canary", json={"pct": 10}, headers=hdr).status_code == 409
        )  # no challenger


# ------------------------------------------------------------------ one command: the demo
@pytest.mark.smoke
@pytest.mark.slow
def test_mlops_demo_end_to_end(tmp_path: Path) -> None:
    from fraudshield.config import Settings
    from fraudshield.mlops.demo import mlops_demo

    out = mlops_demo(Settings(), tmp_path / "demo")
    v1, v2 = out["bootstrap"]["version"], out["challenger"]["version"]
    ch = out["challenger"]
    assert out["bootstrap"]["promoted"] is True
    assert ch["training_set"]["analyst_labels"] > 0
    assert {c["name"] for c in ch["gate"]["checks"]} == {
        "pr_auc",
        "net_savings",
        "latency",
        "parity",
    }
    assert ch["gate"]["shadow"]["decision_agreement"] < 1.0  # the challenger genuinely differs
    # promoted by the gate, or (if the gate blocked it) by a recorded forced promotion
    assert ch["forced"] is (not ch["gate"]["passed"])
    assert out["champion_after_promote"] == v2
    assert out["rollback"] == {"from": v2, "to": v1}
    assert out["champion_after_rollback"] == v1
    assert out["history"] == []
    assert Path(out["challenger"]["report"]).read_text().startswith(f"# Promotion check: v{v2}")


@pytest.mark.smoke
@pytest.mark.slow
def test_prefect_flow_runs_the_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prefect.testing.utilities import prefect_test_harness

    from fraudshield.flows.flows import mlops_demo_flow

    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    with prefect_test_harness():
        out = mlops_demo_flow(workdir=str(tmp_path / "flow-demo"))
    assert out["champion_after_rollback"] == out["bootstrap"]["version"]


@pytest.mark.unit
def test_cli_rollback_without_registry_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commands without --dataset must parse; a missing registry is an error exit, not a traceback."""
    from fraudshield.cli import main
    from fraudshield.config import get_settings

    monkeypatch.setenv(
        "FRAUDSHIELD_TRAINING__MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'empty.db'}"
    )
    get_settings.cache_clear()
    try:
        assert main(["rollback"]) == 1
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_api_loads_champion_and_challenger_from_mlflow(
    registry: tuple[Registry, dict[str, Path], Any], tmp_path: Path
) -> None:
    """Regression for the API's MLflow loader: bundle dirs not named "bundle" must still load."""
    from fraudshield.serving.models import ModelManager

    reg, _, v = registry
    reg.client.set_registered_model_alias("gate-test", "champion", v["weak"])
    reg.client.set_registered_model_alias("gate-test", "challenger", v["strong"])
    s = serving_settings(tmp_path, model_source="mlflow", champion_uri="models:/gate-test@champion",
                         challenger_uri="models:/gate-test@challenger")  # fmt: skip
    s = s.model_copy(
        update={"training": s.training.model_copy(update={"mlflow_tracking_uri": reg.tracking_uri})}
    )
    mm = ModelManager(s)
    assert mm.reload() == {"champion": v["weak"], "challenger": v["strong"]}
    assert mm.champion is not None
    assert mm.champion.bundle.spec.metadata["model_version"] == "synthetic-2x15"
