"""Training smoke test (spec): 10k synthetic rows, 20 trees, full pipeline incl. MLflow registry.

Uses SQLite tracking + a local artifact dir in tmp_path. The numbers asserted are sanity
bounds on a planted synthetic signal, not reported results.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from fraudshield.config import LgbmConfig, PathsConfig, Settings
from fraudshield.model_bundle import ModelBundle
from fraudshield.synthetic import (
    SPLITS,
    synthetic_dataset_config,
    synthetic_features,
    synthetic_settings,
)
from fraudshield.training.model_card import write_model_card
from fraudshield.training.train import TrainResult, prepare_splits, run_training
from fraudshield.training.tune import TuneResult, load_tuned, tune_lgbm

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    return synthetic_features(10_000, seed=0)


def _settings(tmp: Path) -> Settings:
    base = Settings()
    paths = PathsConfig(
        reports=tmp / "reports",
        costs=REPO / "configs/costs.yaml",
        rules=REPO / "configs/rules.yaml",
        tuned_params=tmp / "tuned",
        model_card=tmp / "docs/model_card.md",
    )
    s = synthetic_settings(
        base,
        mlflow_tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
        mlflow_artifact_location=str(tmp / "mlartifacts"),
        categorical_min_count=5,
        lgbm=LgbmConfig(
            num_boost_round=20, early_stopping_rounds=5, num_leaves=15, min_child_samples=20
        ),
    )
    return s.model_copy(update={"paths": paths})


@pytest.fixture(scope="module")
def trained(
    features: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Settings, TrainResult]:
    tmp = tmp_path_factory.mktemp("train")
    s = _settings(tmp)
    result = run_training(
        s, "sparkov", features=features, report_dir=tmp / "reports", bundle_dir=tmp / "bundle"
    )
    return s, result


@pytest.mark.smoke
@pytest.mark.slow
def test_training_run_end_to_end(
    trained: tuple[Settings, TrainResult], features: pl.DataFrame
) -> None:
    _, r = trained
    report = r.report
    models = [row["model"] for row in report["comparison"]]
    assert models == ["rules_only", "logistic", "lgbm_degraded", "lgbm_full"]
    full = report["evals"]["lgbm_full"]["metrics"]
    base_rate = full["base_rate"]
    assert full["pr_auc"] > 3 * base_rate  # the planted signal is learned
    assert report["params"]["num_boost_round"] <= 20
    assert "net_savings_vs_rules_only" in report["evals"]["lgbm_full"]["dollars"]
    assert {"new_uid", "known_uid"} >= set(report["segments"])
    for f in ("comparison.md", "cost_sensitivity.md", "metrics.json", "model_card.md",
              "img/pr_curves.png", "img/shap_summary.png", "img/reliability.png", "img/cost_sensitivity.png"):  # fmt: skip
        assert (r.report_dir / f).is_file(), f
    assert (
        json.loads((r.report_dir / "metrics.json").read_text())["meta"]["model_version"]
        == r.model_version
    )


@pytest.mark.smoke
@pytest.mark.slow
def test_registry_alias_and_pyfunc_match_bundle(
    trained: tuple[Settings, TrainResult], features: pl.DataFrame
) -> None:
    s, r = trained
    mlflow.set_tracking_uri(s.training.mlflow_tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{s.training.registered_model}@challenger")
    bundle = ModelBundle.load(r.bundle_dir)
    test = prepare_splits(features, synthetic_dataset_config())["test"].head(200)
    out = model.predict(test.to_pandas())
    np.testing.assert_allclose(
        out["probability"].to_numpy(), bundle.predict_proba(test), rtol=0, atol=1e-12
    )
    assert set(out["decision"]) <= {"approve", "review", "block"}
    deg = model.predict(test.to_pandas(), params={"variant": "degraded"})
    np.testing.assert_allclose(
        deg["probability"].to_numpy(), bundle.predict_proba(test, "degraded"), atol=1e-12
    )


@pytest.mark.smoke
@pytest.mark.slow
def test_bundle_score_one_matches_batch_and_reasons(
    trained: tuple[Settings, TrainResult], features: pl.DataFrame
) -> None:
    _, r = trained
    bundle = ModelBundle.load(r.bundle_dir)
    test = prepare_splits(features, synthetic_dataset_config())["test"]
    batch = bundle.predict_proba(test.head(50))
    deg_names = set(bundle.spec.degraded.feature_names)
    for i, row in enumerate(test.head(50).to_dicts()):
        p, reasons = bundle.score_one(row)
        assert p == pytest.approx(batch[i], abs=1e-12)
        assert len(reasons) <= 3
        _, deg_reasons = bundle.score_one(row, "degraded")
        assert all(x.feature in deg_names for x in deg_reasons)
    # every bundle artifact is plain text/JSON (no pickles)
    assert sorted(p.name for p in r.bundle_dir.iterdir()) == [
        "bundle.json",
        "degraded.txt",
        "full.txt",
    ]


@pytest.mark.smoke
def test_model_card_renders_from_report(trained: tuple[Settings, TrainResult]) -> None:
    s, r = trained
    out = write_model_card(r.report_dir / "metrics.json", s.paths.model_card)
    text = out.read_text()
    assert text.startswith("# Model card: fraudshield v")
    assert "| lgbm_full |" in text
    assert "## Limitations" in text


@pytest.mark.smoke
@pytest.mark.slow
def test_tune_two_trials_and_persist(features: pl.DataFrame, tmp_path: Path) -> None:
    s = _settings(tmp_path)
    train = prepare_splits(features, synthetic_dataset_config())["train"]
    from fraudshield.pipeline import feature_set_for

    fs = feature_set_for(s, "sparkov")
    res = tune_lgbm(
        train, fs.names, fs.categorical, SPLITS, s.training.lgbm, seed=0, n_trials=2, min_count=5
    )
    assert res.n_trials == 2
    assert 1.0 <= res.params["scale_pos_weight"] <= 50.0
    assert 1 <= res.num_boost_round <= 20
    path = res.save(tmp_path / "tuned.json", fs.spec_hash())
    assert load_tuned(path, fs.spec_hash()) == TuneResult(
        res.params, res.num_boost_round, res.best_value, 2
    )
    assert load_tuned(path, "other-hash") is None
