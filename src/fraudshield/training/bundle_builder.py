"""Assemble a ``ModelBundle`` from a feature frame without MLflow.

Used for synthetic bundles (tests, latency benchmark) and dev bundles trained on the committed
CC0 fixture (so the docker-compose stack has a model to load before the real training run).
None of these are reported models. Real models come from ``training.train.run_training``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from fraudshield.config import LgbmConfig
from fraudshield.features.definitions import FeatureSet
from fraudshield.model_bundle import BundleSpec, ModelBundle, Variant, VariantSpec
from fraudshield.reasons import ReasonMapper
from fraudshield.training.calibrate import Calibrator, fit_calibrator
from fraudshield.training.costs import CostConfig
from fraudshield.training.matrix import CategoryEncoder, build_matrix, typical_values
from fraudshield.training.models import lgbm_params, train_lgbm


def train_bundle(
    fs: FeatureSet,
    train: pl.DataFrame,
    calib: pl.DataFrame,
    costs: CostConfig,
    metadata: dict[str, Any],
    out_dir: Path,
    *,
    lgbm: LgbmConfig,
    rounds: int,
    num_leaves: int,
    seed: int = 0,
    min_count: int = 5,
) -> Path:
    """Train full + degraded boosters on ``train``, calibrate on ``calib``, save the bundle."""
    enc = CategoryEncoder.fit(train, fs.categorical, min_count)
    params = lgbm_params(
        lgbm.model_copy(update={"num_leaves": num_leaves, "min_child_samples": 20}), seed
    )
    boosters: dict[Variant, Any] = {}
    cals: dict[Variant, Calibrator] = {}
    variants: tuple[tuple[Variant, list[str]], ...] = (
        ("full", fs.names),
        ("degraded", fs.request_only),
    )
    for variant, names in variants:
        b = train_lgbm(
            build_matrix(train, names, enc),
            train["label"].to_numpy(),
            names,
            fs.categorical,
            params,
            rounds,
        )
        cals[variant] = fit_calibrator(
            b.predict(build_matrix(calib, names, enc)), calib["label"].to_numpy()
        )
        boosters[variant] = b
    typical = typical_values(train, fs.names, fs.categorical)
    spec = BundleSpec(
        metadata={"feature_set_hash": fs.spec_hash(), **metadata},
        encoder=enc,
        full=VariantSpec(
            booster_file="full.txt", feature_names=tuple(fs.names), calibrator=cals["full"]
        ),
        degraded=VariantSpec(
            booster_file="degraded.txt",
            feature_names=tuple(fs.request_only),
            calibrator=cals["degraded"],
        ),
        costs=costs,
        typical=typical,
        reason_map=ReasonMapper.from_feature_set(fs, fs.names, typical).mapping(),
    )
    return ModelBundle(spec, boosters).save(out_dir)
