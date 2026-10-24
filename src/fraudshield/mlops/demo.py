"""``fraudshield mlops-demo``: the whole loop on synthetic data in one command (Phase 5 DoD).

1. bootstrap: retrain -> v1 @challenger -> gate (no champion) -> v1 @champion
2. retrain with more trees, using analyst labels for recent unmatured chargebacks
   -> v2 @challenger
3. gate v2 vs v1 (PR-AUC, net savings, latency, parity) + shadow comparison -> promote if passed
   (if the gate blocks v2, a recorded ``--force`` promotion follows so step 4 can be shown)
4. rollback -> @champion back to the previous version

Synthetic data and an MLflow registry (SQLite under ``workdir`` unless a tracking URI is given):
it demonstrates the mechanics, not model quality.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudshield.config import LgbmConfig, Settings
from fraudshield.mlops.loop import promote, retrain, rollback
from fraudshield.mlops.registry import Registry
from fraudshield.mlops.training_set import ANALYST_SCHEMA
from fraudshield.synthetic import (
    synthetic_canonical,
    synthetic_dataset_config,
    synthetic_settings,
)
from fraudshield.training.train import prepare_splits

log = logging.getLogger(__name__)


def demo_settings(
    base: Settings,
    workdir: Path,
    tracking_uri: str | None = None,
    model_name: str = "fraudshield-demo",
) -> Settings:
    """SQLite + local artifacts under ``workdir``, or a tracking server (server-side artifacts)."""
    s = synthetic_settings(
        base,
        mlflow_tracking_uri=tracking_uri or f"sqlite:///{(workdir / 'mlflow.db').resolve()}",
        mlflow_artifact_location="" if tracking_uri else str((workdir / "mlartifacts").resolve()),
        registered_model=model_name,
        categorical_min_count=5,
        lgbm=LgbmConfig(
            num_boost_round=20, early_stopping_rounds=10, num_leaves=15, min_child_samples=20
        ),
    )
    paths = s.paths.model_copy(
        update={"reports": workdir / "reports", "tuned_params": workdir / "tuned"}
    )
    return s.model_copy(update={"paths": paths})


def _analyst_labels(features: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Analysts label some recent transactions before their chargebacks mature."""
    last = features["event_time"].max()
    recent = features.filter(pl.col("event_time") > last - timedelta(days=5))  # type: ignore[operator]
    pick = recent.sample(min(n, recent.height), seed=seed)
    rng = np.random.default_rng(seed)
    noisy = np.where(
        rng.random(pick.height) < 0.05, 1 - pick["label"].to_numpy(), pick["label"].to_numpy()
    )
    return pl.DataFrame(
        {"txn_id": pick["txn_id"], "label": noisy.astype(np.int8),
         "label_time": pick["event_time"] + timedelta(hours=6), "analyst_id": ["demo-analyst"] * pick.height},
        schema=ANALYST_SCHEMA,
    )  # fmt: skip


def noisy_features(n: int, label_noise: float, seed: int) -> pl.DataFrame:
    """Synthetic features with ``label_noise`` of labels flipped *before* feature computation.

    The planted signal alone is perfectly separable (PR-AUC 1.0, identical models), which would
    make the gate and shadow comparison vacuous; noise makes the models genuinely differ.
    """
    from fraudshield.features.definitions import build_feature_set
    from fraudshield.features.offline import compute_offline
    from fraudshield.synthetic import EXTRAS

    canon = synthetic_canonical(n, seed=seed)
    flip = pl.Series(np.random.default_rng(seed).random(canon.height) < label_noise)
    canon = canon.with_columns(
        label=pl.when(flip).then(1 - pl.col("label")).otherwise(pl.col("label")).cast(pl.Int8)
    )
    return compute_offline(canon, build_feature_set(EXTRAS, 50.0))


def mlops_demo(
    base: Settings,
    workdir: Path,
    tracking_uri: str | None = None,
    label_noise: float = 0.08,
    model_name: str = "fraudshield-demo",
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    s = demo_settings(base, workdir, tracking_uri, model_name)
    ds = synthetic_dataset_config()
    feats = noisy_features(10_000, label_noise, seed=0)
    test = prepare_splits(feats, ds)["test"]
    canon = synthetic_canonical(2_000, seed=0)
    reg = Registry.from_settings(s)
    out: dict[str, Any] = {
        "tracking_uri": s.training.mlflow_tracking_uri,
        "model": s.training.registered_model,
    }

    def kw(tag: str) -> dict[str, Any]:
        return {
            "ds_cfg": ds,
            "report_dir": workdir / "reports" / tag,
            "bundle_dir": workdir / "bundles" / tag,
        }

    r1, stats1 = retrain(s, "sparkov", features=feats, **kw("v1"))
    p1 = promote(
        s,
        "sparkov",
        test=test,
        parity_canonical=canon,
        ds_cfg=ds,
        registry=reg,
        report_dir=workdir / "promotions",
    )
    out["bootstrap"] = {
        "version": r1.model_version,
        "training_set": stats1,
        "gate_passed": p1.gate.passed,
        "promoted": p1.promoted,
    }

    # v2: different hyperparameters (the analyst labels only touch the most recent days, which fall
    # in the test window under fixed splits, so they alone would not change the trained model)
    lgbm2 = s.training.lgbm.model_copy(update={"num_leaves": 31, "learning_rate": 0.1, "num_boost_round": 60,
                                               "early_stopping_rounds": 20, "feature_fraction": 0.7})  # fmt: skip
    s2 = s.model_copy(update={"training": s.training.model_copy(update={"lgbm": lgbm2})})
    analyst = _analyst_labels(feats, 150, seed=1)
    r2, stats2 = retrain(s2, "sparkov", features=feats, analyst=analyst, **kw("v2"))
    p2 = promote(
        s2,
        "sparkov",
        test=test,
        parity_canonical=canon,
        ds_cfg=ds,
        registry=reg,
        report_dir=workdir / "promotions",
    )
    out["challenger"] = {"version": r2.model_version, "training_set": stats2, "gate": p2.gate.as_dict(),
                         "promoted": p2.promoted, "forced": False, "report": str(p2.report)}  # fmt: skip
    if not p2.promoted:
        # The gate blocked v2. To still demonstrate rollback, do what an operator would after
        # overriding the gate: a recorded forced promotion, then roll back.
        pf = promote(s2, "sparkov", test=test, parity_canonical=canon, ds_cfg=ds, registry=reg,
                     report_dir=workdir / "promotions", force=True)  # fmt: skip
        out["challenger"] |= {"forced": True, "forced_report": str(pf.report)}
    out["champion_after_promote"] = reg.version_of(reg.champion)
    move = rollback(s2, registry=reg)
    out["rollback"] = {"from": move.before, "to": move.after}
    out["champion_after_rollback"] = reg.version_of(reg.champion)
    out["history"] = reg.history()
    return out
