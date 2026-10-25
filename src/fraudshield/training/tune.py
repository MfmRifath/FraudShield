"""Optuna tuning of the full LightGBM model with time-aware (rolling-origin) CV inside train.

Objective: mean validation PR-AUC over the folds of ``splits.rolling_origin_folds``. Each fold
fits its own category vocabulary on its own training days, so no fold sees later categories.
``scale_pos_weight`` is a searched hyperparameter; there is no resampling.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
from sklearn.metrics import average_precision_score

from fraudshield.config import LgbmConfig, SplitConfig
from fraudshield.training.matrix import CategoryEncoder, build_matrix
from fraudshield.training.models import lgbm_params, train_lgbm
from fraudshield.training.splits import rolling_origin_folds

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuneResult:
    params: dict[str, Any]
    num_boost_round: int
    best_value: float
    n_trials: int

    def save(self, path: Path, feature_set_hash: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "cv_mean_pr_auc": self.best_value,
            "n_trials": self.n_trials,
            "feature_set_hash": feature_set_hash,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


def load_tuned(path: Path, feature_set_hash: str) -> TuneResult | None:
    """Tuned params, or None if absent or tuned for a different feature set."""
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    if d.get("feature_set_hash") != feature_set_hash:
        log.warning("ignoring %s: tuned for a different feature set", path)
        return None
    return TuneResult(
        d["params"], int(d["num_boost_round"]), float(d["cv_mean_pr_auc"]), int(d["n_trials"])
    )


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
    }


def tune_lgbm(
    train: pl.DataFrame,
    feature_names: Sequence[str],
    categorical: Sequence[str],
    splits: SplitConfig,
    base: LgbmConfig,
    seed: int,
    n_trials: int,
    min_count: int,
    timeout_s: int | None = None,
) -> TuneResult:
    """``train`` holds labelled training-window rows with a ``day`` column."""
    folds = []
    for fold in rolling_origin_folds(splits):
        tr = train.filter(pl.col("day").is_between(fold.train.start, fold.train.end))
        va = train.filter(pl.col("day").is_between(fold.val.start, fold.val.end))
        enc = CategoryEncoder.fit(tr, categorical, min_count)
        folds.append(
            (
                build_matrix(tr, feature_names, enc),
                tr["label"].to_numpy(),
                build_matrix(va, feature_names, enc),
                va["label"].to_numpy(),
            )
        )

    def objective(trial: optuna.Trial) -> float:
        params = lgbm_params(base, seed, search_space(trial))
        scores, rounds = [], []
        for x_tr, y_tr, x_va, y_va in folds:
            booster = train_lgbm(
                x_tr, y_tr, feature_names, categorical, params, base.num_boost_round,
                valid=(x_va, y_va), early_stopping_rounds=base.early_stopping_rounds,
            )  # fmt: skip
            p = booster.predict(x_va, num_iteration=booster.best_iteration)
            scores.append(average_precision_score(y_va, p) if 0 < y_va.sum() < y_va.size else 0.0)
            rounds.append(booster.best_iteration or base.num_boost_round)
        trial.set_user_attr("num_boost_round", int(np.mean(rounds)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    started = time.monotonic()

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # one line per trial: a long tuning run otherwise prints nothing for hours
        elapsed = time.monotonic() - started
        done = len(study.trials)
        eta = elapsed / done * (n_trials - done) if done else 0.0
        log.info("trial %d/%d: CV PR-AUC %.4f (best %.4f), %.0f min elapsed, ~%.0f min left",
                 done, n_trials, trial.value or 0.0, study.best_value, elapsed / 60, eta / 60)  # fmt: skip

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, callbacks=[progress])
    best = study.best_trial
    return TuneResult(
        params=dict(best.params),
        num_boost_round=int(best.user_attrs["num_boost_round"]),
        best_value=float(best.value or 0.0),
        n_trials=len(study.trials),
    )
