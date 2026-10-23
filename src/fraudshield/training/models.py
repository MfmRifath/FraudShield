"""Model trainers: LightGBM (full and request-only "degraded") and a logistic-regression baseline.

Class imbalance is handled with ``scale_pos_weight`` (tuned), never by resampling: SMOTE and
friends distort calibration and the temporal structure the features rely on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudshield.config import LgbmConfig

FloatArray = npt.NDArray[np.float64]


def lgbm_params(
    cfg: LgbmConfig, seed: int, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "scale_pos_weight": cfg.scale_pos_weight,
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if cfg.num_threads:
        params["num_threads"] = cfg.num_threads
    params.update(overrides or {})
    return params


def train_lgbm(
    x: FloatArray,
    y: npt.ArrayLike,
    feature_names: Sequence[str],
    categorical: Sequence[str],
    params: Mapping[str, Any],
    num_boost_round: int,
    valid: tuple[FloatArray, npt.ArrayLike] | None = None,
    early_stopping_rounds: int | None = None,
) -> lgb.Booster:
    """Train one booster; with ``valid`` + ``early_stopping_rounds`` it stops at the best round."""
    cats = [c for c in categorical if c in feature_names]
    train_set = lgb.Dataset(
        x,
        label=np.asarray(y),
        feature_name=list(feature_names),
        categorical_feature=cats,
        free_raw_data=False,
    )
    valid_sets: list[lgb.Dataset] = []
    callbacks: list[Callable[..., Any]] = []
    if valid is not None:
        valid_sets = [lgb.Dataset(valid[0], label=np.asarray(valid[1]), reference=train_set)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(
        dict(params),
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


class LogisticBaseline:
    """Median-impute (+ missing indicators) and scale numerics; one-hot categorical codes."""

    def __init__(self, feature_names: Sequence[str], categorical: Sequence[str], seed: int) -> None:
        names = list(feature_names)
        cat_idx = [i for i, n in enumerate(names) if n in categorical]
        num_idx = [i for i, n in enumerate(names) if n not in categorical]
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        cat = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="constant", fill_value=-1.0, keep_empty_features=True),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("prep", ColumnTransformer([("num", numeric, num_idx), ("cat", cat, cat_idx)])),
                (
                    "lr",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        )

    def fit(self, x: FloatArray, y: npt.ArrayLike) -> LogisticBaseline:
        self.pipeline.fit(x, np.asarray(y))
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(self.pipeline.predict_proba(x)[:, 1], dtype=np.float64)
        return out
