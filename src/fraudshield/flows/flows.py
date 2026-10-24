"""Prefect 3 flows (thin wrappers over ``fraudshield.pipeline`` / ``fraudshield.mlops``).

Run one locally: ``python -m fraudshield.flows run retrain-and-promote``.
Serve them (prefect-worker container): ``python -m fraudshield.flows serve``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from prefect import flow, get_run_logger, task

from fraudshield.config import DatasetName, Settings, get_settings
from fraudshield.mlops.training_set import ANALYST_SCHEMA, build_training_set
from fraudshield.pipeline import DatasetPaths, build_features


def _settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


def _dataset(settings: Settings, dataset: str | None) -> DatasetName:
    name = dataset or settings.dataset
    if name not in ("ieee_cis", "sparkov"):
        raise ValueError(f"unknown dataset {name}")
    return name  # type: ignore[return-value]


def load_analyst_labels(settings: Settings) -> pl.DataFrame:
    """Labelled review cases exported to Parquet (the Postgres case store replaces this in Phase 7)."""
    path = settings.paths.analyst_labels
    return pl.read_parquet(path) if path.is_file() else pl.DataFrame(schema=ANALYST_SCHEMA)


@flow(name="backfill-features")
def backfill_features(dataset: str | None = None, check: bool = False) -> dict[str, Any]:
    s = _settings()
    m = build_features(s, _dataset(s, dataset), check=check)
    return {"rows": m.rows, "features_sha256": m.features_sha256}


@flow(name="build-training-set")
def build_training_set_flow(
    dataset: str | None = None, as_of: datetime | None = None
) -> dict[str, Any]:
    s = _settings()
    ds = _dataset(s, dataset)
    paths = DatasetPaths.of(s, ds)
    feats = pl.read_parquet(paths.features)
    last_event: datetime = feats["event_time"].max()  # type: ignore[assignment]
    cutoff = as_of if as_of is not None else last_event
    out, stats = build_training_set(feats, load_analyst_labels(s), cutoff)
    target = paths.features.with_name("training_set.parquet")
    out.write_parquet(target, compression="zstd")
    get_run_logger().info("training set %s: %s", target, stats.as_dict())
    return stats.as_dict() | {"path": str(target)}


@task(name="retrain")
def _retrain(dataset: DatasetName, as_of: datetime | None) -> str | None:
    from fraudshield.mlops.loop import retrain

    s = _settings()
    result, _ = retrain(s, dataset, analyst=load_analyst_labels(s), as_of=as_of)
    return result.model_version


@task(name="promote")
def _promote(dataset: DatasetName, force: bool) -> dict[str, Any]:
    from fraudshield.mlops.loop import promote

    out = promote(_settings(), dataset, force=force)
    return out.gate.as_dict() | {
        "promoted": out.promoted,
        "challenger": out.challenger,
        "report": str(out.report),
    }


@flow(name="retrain")
def retrain_flow(dataset: str | None = None, as_of: datetime | None = None) -> str | None:
    s = _settings()
    return _retrain(_dataset(s, dataset), as_of)


@flow(name="promote")
def promote_flow(dataset: str | None = None, force: bool = False) -> dict[str, Any]:
    s = _settings()
    return _promote(_dataset(s, dataset), force)


@flow(name="retrain-and-promote")
def retrain_and_promote(
    dataset: str | None = None, as_of: datetime | None = None
) -> dict[str, Any]:
    """Retrain -> @challenger -> gate (incl. shadow comparison) -> promote if every check passes."""
    s = _settings()
    ds = _dataset(s, dataset)
    version = _retrain(ds, as_of)
    result = _promote(ds, False)
    get_run_logger().info(
        "v%s: gate %s, promoted=%s", version, result["passed"], result["promoted"]
    )
    return {"version": version, **result}


@flow(name="mlops-demo")
def mlops_demo_flow(
    workdir: str = "artifacts/mlops-demo",
    tracking_uri: str | None = None,
    model_name: str = "fraudshield-demo",
) -> dict[str, Any]:
    from fraudshield.mlops.demo import mlops_demo

    s = _settings()
    uri = s.training.mlflow_tracking_uri
    # on the compose stack the worker tracks to the MLflow server; locally the demo uses SQLite
    server = uri if uri.startswith(("http://", "https://")) else None
    return mlops_demo(s, Path(workdir), tracking_uri or server, model_name=model_name)
