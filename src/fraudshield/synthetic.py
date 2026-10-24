"""Synthetic transactions with a planted fraud signal (smoke tests, latency benchmark).

Not a benchmark: the signal (compromised cards making bursts of large transactions on a shared
device) is deliberately easy, so tests can assert the pipeline *learns something*.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from fraudshield.adapters.base import finalize_canonical
from fraudshield.config import DatasetConfig, DayRange, LabelDelayConfig, Settings, SplitConfig
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline

ORIGIN = datetime(2020, 12, 31, tzinfo=UTC)  # day 1 = 2021-01-01
N_DAYS = 70
EXTRAS = ("C1",)
SPLITS = SplitConfig(
    train=DayRange(start=1, end=40),
    val=DayRange(start=48, end=55),
    test=DayRange(start=63, end=70),
    gap_days=7,
    cv_folds=2,
    cv_val_days=7,
    cv_min_train_days=10,
)


def synthetic_dataset_config() -> DatasetConfig:
    return DatasetConfig(
        kaggle_handle="synthetic/none",
        kaggle_kind="dataset",
        license="synthetic",
        origin=ORIGIN,
        splits=SPLITS,
        extra_columns=EXTRAS,
    )


def synthetic_settings(base: Settings, **training_overrides: Any) -> Settings:
    """``base`` with the sparkov slot replaced by the synthetic dataset config."""
    datasets = base.datasets.model_copy(update={"sparkov": synthetic_dataset_config()})
    training = base.training.model_copy(update=training_overrides)
    return base.model_copy(update={"datasets": datasets, "training": training})


def synthetic_canonical(n: int = 10_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_uids = 400
    uids = [f"u{i:03d}" for i in range(n_uids)]
    home_device = {u: f"dev{rng.integers(0, 600):03d}" for u in uids}
    home_email = {
        u: str(rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anon.com"])) for u in uids
    }
    compromised = set(rng.choice(uids, size=40, replace=False).tolist())
    rows: list[dict[str, Any]] = []
    n_fraud_target = int(n * 0.08)
    # fraud bursts: 3-8 large transactions within ~20 minutes on a shared "ring" device
    while sum(r["label"] for r in rows) < n_fraud_target:
        u = str(rng.choice(sorted(compromised)))
        start = ORIGIN + timedelta(days=float(rng.uniform(1, N_DAYS + 1)))
        for _ in range(int(rng.integers(3, 9))):
            rows.append(
                _row(u, start + timedelta(seconds=float(rng.uniform(0, 1200))),
                     float(np.round(rng.lognormal(6.0, 0.5), 2)), f"ring{rng.integers(0, 3)}",
                     "anon.com", "W", 1, rng)
            )  # fmt: skip
    while len(rows) < n:
        u = uids[int(rng.zipf(1.3)) % n_uids]
        t = ORIGIN + timedelta(days=float(rng.uniform(1, N_DAYS + 1)))
        dev = home_device[u] if rng.random() < 0.9 else f"dev{rng.integers(0, 600):03d}"
        rows.append(
            _row(u, t, float(np.round(rng.lognormal(3.5, 1.0), 2)), dev, home_email[u],
                 str(rng.choice(["W", "C", "H"])), 0, rng)
        )  # fmt: skip
    schema: dict[str, pl.DataType] = {
        "txn_id": pl.String(), "event_time": pl.Datetime("us", "UTC"), "uid": pl.String(),
        "amount": pl.Float64(), "product": pl.String(), "card_network": pl.String(),
        "p_email_domain": pl.String(), "device_type": pl.String(), "device_info": pl.String(),
        "addr": pl.String(), "label": pl.Int8(), "x_c1": pl.Float64(),
    }  # fmt: skip
    df = pl.DataFrame(rows[:n], schema=schema)
    delay = LabelDelayConfig(kind="uniform", min_days=1.0, max_days=7.0, seed=1)
    return finalize_canonical(df, EXTRAS, delay)


def _row(
    uid: str, t: datetime, amount: float, device: str, email: str, product: str, label: int,
    rng: np.random.Generator,
) -> dict[str, Any]:  # fmt: skip
    return {
        "txn_id": f"s{rng.integers(0, 2**62):x}",
        "event_time": t.replace(microsecond=0),
        "uid": uid,
        "amount": amount,
        "product": product,
        "card_network": "visa" if int(uid[1:]) % 2 else "mastercard",
        "p_email_domain": email,
        "device_type": "mobile" if device.startswith("ring") else "desktop",
        "device_info": device,
        "addr": str(int(uid[1:]) % 50),
        "label": label,
        "x_c1": float(rng.poisson(2)),
    }


def synthetic_features(n: int = 10_000, seed: int = 0, te_smoothing: float = 50.0) -> pl.DataFrame:
    return compute_offline(synthetic_canonical(n, seed), build_feature_set(EXTRAS, te_smoothing))


def build_synthetic_bundle(
    settings: Settings,
    out_dir: Any,
    *,
    features: pl.DataFrame | None = None,
    rounds: int = 20,
    num_leaves: int = 15,
    seed: int = 0,
    label_noise: float = 0.0,
) -> Any:
    """Train full + degraded boosters on synthetic data and save a ModelBundle (no MLflow).

    ``settings`` must come from ``synthetic_settings`` so the feature set matches. Used by API
    tests (small) and the latency benchmark (production-sized trees); never a reported model.
    ``label_noise`` flips that share of training labels: the planted signal is so easy that
    LightGBM otherwise stops splitting (stumps), which would make a latency benchmark optimistic.
    """
    from pathlib import Path

    from fraudshield.model_bundle import BundleSpec, ModelBundle, VariantSpec
    from fraudshield.model_bundle import Variant as BundleVariant
    from fraudshield.pipeline import feature_set_for
    from fraudshield.reasons import ReasonMapper
    from fraudshield.training.calibrate import fit_calibrator
    from fraudshield.training.costs import CostConfig
    from fraudshield.training.matrix import CategoryEncoder, build_matrix, typical_values
    from fraudshield.training.models import lgbm_params, train_lgbm
    from fraudshield.training.splits import assign_splits

    fs = feature_set_for(settings, "sparkov")
    feats = features if features is not None else synthetic_features(te_smoothing=fs.te_smoothing)
    df = assign_splits(feats, ORIGIN, SPLITS).filter(pl.col("label").is_not_null())
    if label_noise > 0:
        flip = np.random.default_rng(seed).random(df.height) < label_noise
        df = df.with_columns(
            label=pl.when(pl.Series(flip)).then(1 - pl.col("label")).otherwise(pl.col("label"))
        )
    tr, va = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "val")
    enc = CategoryEncoder.fit(tr, fs.categorical, 5)
    cfg = settings.training.lgbm.model_copy(
        update={"num_leaves": num_leaves, "min_child_samples": 20}
    )
    params = lgbm_params(cfg, seed)
    boosters: dict[BundleVariant, Any] = {}
    cals: dict[BundleVariant, Any] = {}
    variants: tuple[tuple[BundleVariant, list[str]], ...] = (
        ("full", fs.names),
        ("degraded", fs.request_only),
    )
    for variant, names in variants:
        b = train_lgbm(
            build_matrix(tr, names, enc),
            tr["label"].to_numpy(),
            names,
            fs.categorical,
            params,
            rounds,
        )
        cals[variant] = fit_calibrator(
            b.predict(build_matrix(va, names, enc)), va["label"].to_numpy()
        )
        boosters[variant] = b
    typical = typical_values(tr, fs.names, fs.categorical)
    spec = BundleSpec(
        metadata={
            "dataset": "synthetic",
            "feature_set_hash": fs.spec_hash(),
            "created_at": "synthetic",
            "model_version": f"synthetic-{rounds}x{num_leaves}",
        },
        encoder=enc,
        full=VariantSpec(
            booster_file="full.txt", feature_names=tuple(fs.names), calibrator=cals["full"]
        ),
        degraded=VariantSpec(
            booster_file="degraded.txt",
            feature_names=tuple(fs.request_only),
            calibrator=cals["degraded"],
        ),
        costs=CostConfig.load(settings.paths.costs),
        typical=typical,
        reason_map=ReasonMapper.from_feature_set(fs, fs.names, typical).mapping(),
    )
    return ModelBundle(spec, boosters).save(Path(out_dir))
