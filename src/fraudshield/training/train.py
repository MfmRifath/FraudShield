"""Training run: baselines + LightGBM, calibration, evaluation, MLflow logging, packaging.

Models compared on the test window (same policy, costs and capacity for all):

* ``rules_only``: ``configs/rules.yaml`` alone (block/review/approve);
* ``logistic``: logistic regression on all features;
* ``lgbm_degraded``: LightGBM on request-only features (the serving fallback);
* ``lgbm_full``: LightGBM on all features (the model that gets registered).

Boosting rounds come from ``configs/tuned/<ds>.json`` (``make tune``) when present; otherwise
from early stopping on the last rolling-origin fold inside train. Final models are refit on
the whole train window, calibrated on val and evaluated once on test.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from fraudshield.config import DatasetConfig, DatasetName, Settings
from fraudshield.features.definitions import FeatureSet
from fraudshield.manifest import FeaturesManifest, read_json
from fraudshield.model_bundle import BundleSpec, ModelBundle, VariantSpec
from fraudshield.pipeline import DatasetPaths, feature_set_for
from fraudshield.reasons import ReasonMapper
from fraudshield.rules import RuleSet
from fraudshield.training.calibrate import fit_calibrator
from fraudshield.training.costs import DECISION_DTYPE, CostConfig, decide, limit_reviews
from fraudshield.training.evaluate import (
    ModelEval,
    add_savings_vs,
    comparison_table,
    cost_sensitivity,
    evaluate,
    plot_cost_sensitivity,
    plot_pr_curves,
    plot_reliability,
    to_markdown,
)
from fraudshield.training.matrix import CategoryEncoder, build_matrix, typical_values
from fraudshield.training.model_card import render_model_card
from fraudshield.training.models import LogisticBaseline, lgbm_params, train_lgbm
from fraudshield.training.package import log_and_register
from fraudshield.training.splits import SPLITS, assign_splits, rolling_origin_folds
from fraudshield.training.tune import load_tuned

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_version: str | None
    bundle_dir: Path
    report_dir: Path
    report: dict[str, Any]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=5,
        )  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def prepare_splits(features: pl.DataFrame, ds: DatasetConfig) -> dict[str, pl.DataFrame]:
    """Labelled rows per split, with ``day`` and ``is_new_uid`` (no earlier event for the uid)."""
    df = (
        assign_splits(features, ds.origin, ds.splits)
        .filter(pl.col("label").is_not_null() & pl.col("split").is_not_null())
        .with_columns(is_new_uid=pl.col("uid_secs_since_last").is_null())
    )
    out: dict[str, pl.DataFrame] = {s: df.filter(pl.col("split") == s) for s in SPLITS}
    for s, f in out.items():
        if f.is_empty() or f["label"].n_unique() < 2:
            raise ValueError(f"{s} split needs both classes (rows={f.height})")
    return out


def _rounds_by_early_stopping(
    train: pl.DataFrame,
    names: list[str],
    fs: FeatureSet,
    ds: DatasetConfig,
    params: dict[str, Any],
    settings: Settings,
) -> int:
    """Best round on the last rolling-origin fold (train-window only; val/test untouched)."""
    lg = settings.training.lgbm
    fold = rolling_origin_folds(ds.splits)[-1]
    tr = train.filter(pl.col("day").is_between(fold.train.start, fold.train.end))
    va = train.filter(pl.col("day").is_between(fold.val.start, fold.val.end))
    if tr["label"].n_unique() < 2 or va["label"].n_unique() < 2:
        return lg.num_boost_round
    enc = CategoryEncoder.fit(tr, fs.categorical, settings.training.categorical_min_count)
    booster = train_lgbm(
        build_matrix(tr, names, enc), tr["label"].to_numpy(), names, fs.categorical, params,
        lg.num_boost_round, valid=(build_matrix(va, names, enc), va["label"].to_numpy()),
        early_stopping_rounds=lg.early_stopping_rounds,
    )  # fmt: skip
    return int(booster.best_iteration or lg.num_boost_round)


def _rules_only(
    rules: RuleSet, test: pl.DataFrame, costs: CostConfig
) -> tuple[np.ndarray, np.ndarray]:
    action = (
        rules.evaluate_frame(test)["rule_action"]
        .fill_null("approve")
        .to_numpy()
        .astype(DECISION_DTYPE)
    )
    approve = np.full(action.shape, "approve", dtype=DECISION_DTYPE)
    decisions = limit_reviews(
        action,
        test["amount"].to_numpy(),
        test["day"].to_numpy(),
        costs.review_capacity_per_day,
        approve,
    )
    score = np.select([action == "block", action == "review"], [1.0, 0.5], 0.0)
    return score, decisions


def _setup_mlflow(settings: Settings) -> None:
    t = settings.training
    mlflow.set_tracking_uri(t.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(t.experiment) is None:
        location = t.mlflow_artifact_location
        if "://" not in location:
            location = Path(location).resolve().as_uri()
        mlflow.create_experiment(t.experiment, artifact_location=location)
    mlflow.set_experiment(t.experiment)


def run_training(
    settings: Settings,
    dataset: DatasetName,
    *,
    features: pl.DataFrame | None = None,
    ds_cfg: DatasetConfig | None = None,
    register: bool = True,
    report_dir: Path | None = None,
    bundle_dir: Path | None = None,
) -> TrainResult:
    ds = ds_cfg or settings.datasets.get(dataset)
    paths = DatasetPaths.of(settings, dataset)
    if features is None:
        features = pl.read_parquet(paths.features)
        manifest = read_json(FeaturesManifest, paths.features_manifest)
        raw_hash, feat_sha = manifest.raw_manifest_hash, manifest.features_sha256
    else:
        raw_hash, feat_sha = "in-memory", "in-memory"
    report_dir = report_dir or settings.paths.reports / dataset
    bundle_dir = bundle_dir or Path("artifacts") / dataset / "bundle"

    fs = feature_set_for(settings, dataset)
    names, request, categorical = fs.names, fs.request_only, fs.categorical
    frames = prepare_splits(features, ds)
    tr, te = frames["train"], frames["test"]
    t = settings.training
    seed = settings.seed

    encoder = CategoryEncoder.fit(tr, categorical, t.categorical_min_count)
    x = {s: build_matrix(f, names, encoder) for s, f in frames.items()}
    xr = {s: build_matrix(f, request, encoder) for s, f in frames.items()}
    y = {s: f["label"].to_numpy() for s, f in frames.items()}

    tuned = load_tuned(settings.paths.tuned_params / f"{dataset}.json", fs.spec_hash())
    params_full = lgbm_params(t.lgbm, seed, tuned.params if tuned else None)
    params_deg = lgbm_params(t.lgbm, seed)
    rounds_full = (
        tuned.num_boost_round
        if tuned
        else _rounds_by_early_stopping(tr, names, fs, ds, params_full, settings)
    )
    rounds_deg = _rounds_by_early_stopping(tr, request, fs, ds, params_deg, settings)
    log.info("training full (%d rounds) and degraded (%d rounds) boosters", rounds_full, rounds_deg)
    full = train_lgbm(x["train"], y["train"], names, categorical, params_full, rounds_full)
    degraded = train_lgbm(xr["train"], y["train"], request, categorical, params_deg, rounds_deg)
    logistic = LogisticBaseline(names, categorical, seed).fit(x["train"], y["train"])

    def raw(
        model: lgb.Booster | LogisticBaseline, split: str, cols: dict[str, np.ndarray]
    ) -> np.ndarray:
        if isinstance(model, lgb.Booster):
            return np.asarray(model.predict(cols[split]), dtype=np.float64)
        return model.predict_proba(cols[split])

    calibrators = {
        name: fit_calibrator(raw(m, "val", cols), y["val"], t.platt_below_positives)
        for name, m, cols in (
            ("lgbm_full", full, x),
            ("lgbm_degraded", degraded, xr),
            ("logistic", logistic, x),
        )
    }
    test_p = {
        name: calibrators[name].apply(raw(m, "test", cols))
        for name, m, cols in (
            ("lgbm_full", full, x),
            ("lgbm_degraded", degraded, xr),
            ("logistic", logistic, x),
        )
    }

    costs = CostConfig.load(settings.paths.costs)
    rules = RuleSet.load(settings.paths.rules)
    amount, day = te["amount"].to_numpy(), te["day"].to_numpy()
    rule_score, rule_decisions = _rules_only(rules, te, costs)
    evals: list[ModelEval] = [evaluate("rules_only", te, rule_score, rule_decisions, costs)]
    for name in ("logistic", "lgbm_degraded", "lgbm_full"):
        evals.append(
            evaluate(name, te, test_p[name], decide(test_p[name], amount, day, costs), costs)
        )
    add_savings_vs(evals, "rules_only")
    comparison = comparison_table(evals)
    sweep = cost_sensitivity(te, test_p["lgbm_full"], costs)

    report_dir.mkdir(parents=True, exist_ok=True)
    img = report_dir / "img"
    curves = {n: (y["test"], p) for n, p in test_p.items()}
    plots = [
        plot_pr_curves(curves | {"rules_only": (y["test"], rule_score)}, img / "pr_curves.png"),
        plot_reliability(curves, img / "reliability.png"),
        plot_cost_sensitivity(sweep, img / "cost_sensitivity.png"),
    ]
    (report_dir / "comparison.md").write_text(to_markdown(comparison))
    (report_dir / "cost_sensitivity.md").write_text(to_markdown(sweep))

    typical = typical_values(tr, names, categorical)
    reason_map = ReasonMapper.from_feature_set(fs, names, typical).mapping()
    sha = git_sha()
    meta: dict[str, Any] = {
        "dataset": dataset,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "feature_set_hash": fs.spec_hash(),
        "raw_manifest_hash": raw_hash,
        "features_sha256": feat_sha,
        "registered_model": t.registered_model,
        "versions": {p: version(p) for p in ("fraudshield", "lightgbm", "polars", "mlflow")},
    }
    spec = BundleSpec(
        metadata=meta,
        encoder=encoder,
        full=VariantSpec(
            booster_file="full.txt", feature_names=tuple(names), calibrator=calibrators["lgbm_full"]
        ),
        degraded=VariantSpec(
            booster_file="degraded.txt",
            feature_names=tuple(request),
            calibrator=calibrators["lgbm_degraded"],
        ),
        costs=costs,
        typical=typical,
        reason_map=reason_map,
    )
    ModelBundle(spec, {"full": full, "degraded": degraded}).save(bundle_dir)

    _setup_mlflow(settings)
    with mlflow.start_run(run_name=f"train-{dataset}") as run:
        mlflow.set_tags({"dataset": dataset, "git_sha": sha, "feature_set_hash": fs.spec_hash(),
                         "raw_manifest_hash": raw_hash, "features_sha256": feat_sha,
                         "tuned": str(tuned is not None)})  # fmt: skip
        mlflow.log_params({f"lgbm.{k}": v for k, v in params_full.items()} | {
            "rounds_full": rounds_full, "rounds_degraded": rounds_deg, "n_features": len(names),
            "n_request_features": len(request), **{f"cost.{k}": v for k, v in costs.model_dump().items()},
            **{f"split.{s}": f"{getattr(ds.splits, s).start}-{getattr(ds.splits, s).end}" for s in SPLITS},
        })  # fmt: skip
        for e in evals:
            mlflow.log_metrics({f"{e.name}.{k}": v for k, v in (e.metrics | e.dollars).items()})
        for p in plots:
            mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_artifact(str(report_dir / "comparison.md"))
        mlflow.log_artifact(str(report_dir / "cost_sensitivity.md"))
        model_version = None
        if register:
            _, model_version = log_and_register(
                bundle_dir, te, t.registered_model, alias="challenger"
            )
        run_id = run.info.run_id

        full_eval = next(e for e in evals if e.name == "lgbm_full")
        report: dict[str, Any] = {
            "meta": meta | {"run_id": run_id, "model_version": model_version},
            "data": {
                "gap_days": ds.splits.gap_days,
                "splits": {s: getattr(ds.splits, s).model_dump() for s in SPLITS},
                "rows": {s: f.height for s, f in frames.items()},
                "positives": {s: int(f["label"].sum()) for s, f in frames.items()},
            },
            "features": {"n_full": len(names), "n_request_only": len(request)},
            "params": {k: v for k, v in params_full.items() if k not in ("verbosity",)}
            | {"num_boost_round": rounds_full},
            "calibration": {
                "full": calibrators["lgbm_full"].kind,
                "degraded": calibrators["lgbm_degraded"].kind,
            },
            "costs": costs.model_dump(),
            "comparison": comparison.to_dicts(),
            "segments": full_eval.segments,
            "evals": {e.name: {"metrics": e.metrics, "dollars": e.dollars} for e in evals},
        }
        (report_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        (report_dir / "model_card.md").write_text(
            render_model_card(report | {"report_path": str(report_dir / "metrics.json")})
        )
        mlflow.log_artifact(str(report_dir / "metrics.json"))
        mlflow.log_artifact(str(report_dir / "model_card.md"))
    log.info("run %s: model version %s, report %s", run_id, model_version, report_dir)
    return TrainResult(run_id, model_version, bundle_dir, report_dir, report)
