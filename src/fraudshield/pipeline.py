"""Batch data pipeline: raw files -> canonical parquet (interim) -> features parquet (processed)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import polars as pl

from fraudshield.adapters import DatasetAdapter, get_adapter
from fraudshield.config import DatasetName, Settings
from fraudshield.download import download
from fraudshield.features.definitions import FeatureSet, build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.manifest import (
    FeaturesManifest,
    RawManifest,
    frame_content_sha256,
    read_json,
    record_raw,
    verify_raw,
    write_json,
)

log = logging.getLogger(__name__)


class ReproducibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    canonical: Path
    features: Path
    raw_manifest: Path
    features_manifest: Path

    @classmethod
    def of(cls, settings: Settings, dataset: DatasetName) -> DatasetPaths:
        p = settings.paths
        return cls(
            raw_dir=p.data_raw / dataset,
            canonical=p.data_interim / dataset / "transactions.parquet",
            features=p.data_processed / dataset / "features.parquet",
            raw_manifest=p.manifests / f"{dataset}.raw.json",
            features_manifest=p.manifests / f"{dataset}.features.json",
        )


def feature_set_for(settings: Settings, dataset: DatasetName) -> FeatureSet:
    return build_feature_set(
        settings.datasets.get(dataset).extra_columns, settings.features.target_encoding_smoothing
    )


def fetch_raw(settings: Settings, dataset: DatasetName, force: bool = False) -> RawManifest:
    """Download (if needed) and verify against / record the committed raw manifest."""
    cfg = settings.datasets.get(dataset)
    adapter = get_adapter(dataset, settings)
    paths = DatasetPaths.of(settings, dataset)
    download(cfg, paths.raw_dir, list(adapter.raw_files), force=force)
    if paths.raw_manifest.is_file():
        manifest = read_json(RawManifest, paths.raw_manifest)
        problems = verify_raw(manifest, paths.raw_dir)
        if problems:
            raise ReproducibilityError(
                f"raw files differ from {paths.raw_manifest}: {problems}. "
                "If upstream data changed intentionally, delete the manifest and re-run."
            )
        log.info("raw files match %s", paths.raw_manifest)
        return manifest
    source = f"kaggle:{cfg.kaggle_kind}:{cfg.kaggle_handle}"
    manifest = record_raw(dataset, source, cfg.license, adapter.default_files(paths.raw_dir))
    write_json(manifest, paths.raw_manifest)
    log.info("recorded new raw manifest %s", paths.raw_manifest)
    return manifest


def build_tables(
    adapter: DatasetAdapter, files: list[Path], fs: FeatureSet
) -> tuple[pl.DataFrame, pl.DataFrame]:
    canonical = adapter.load(files)
    return canonical, compute_offline(canonical, fs)


def build_features(
    settings: Settings, dataset: DatasetName, check: bool = False
) -> FeaturesManifest:
    """Rebuild canonical + feature parquet files and their manifest.

    With ``check=True`` the freshly computed content hashes must equal the committed manifest
    (raises ``ReproducibilityError`` otherwise) and nothing is overwritten.
    """
    paths = DatasetPaths.of(settings, dataset)
    if not paths.raw_manifest.is_file():
        raise FileNotFoundError(
            f"{paths.raw_manifest} not found: run `make data DATASET={dataset}`"
        )
    raw = read_json(RawManifest, paths.raw_manifest)
    problems = verify_raw(raw, paths.raw_dir)
    if problems:
        raise ReproducibilityError(f"raw files differ from manifest: {problems}")

    adapter = get_adapter(dataset, settings)
    fs = feature_set_for(settings, dataset)
    canonical, features = build_tables(adapter, adapter.default_files(paths.raw_dir), fs)
    manifest = FeaturesManifest(
        dataset=dataset,
        raw_manifest_hash=raw.content_hash(),
        feature_set_hash=fs.spec_hash(),
        label_delay=settings.labels.delay.model_dump(),
        feature_names=tuple(fs.names),
        rows=features.height,
        canonical_sha256=frame_content_sha256(canonical),
        features_sha256=frame_content_sha256(features),
        versions={p: version(p) for p in ("fraudshield", "polars", "duckdb", "pyarrow")},
    )
    if check:
        if not paths.features_manifest.is_file():
            raise FileNotFoundError(f"{paths.features_manifest} not found: run `make features`")
        expected = read_json(FeaturesManifest, paths.features_manifest)
        diffs = {
            k: (getattr(expected, k), getattr(manifest, k))
            for k in FeaturesManifest.model_fields
            if k != "versions" and getattr(expected, k) != getattr(manifest, k)
        }
        if expected.versions != manifest.versions:
            log.warning("library versions differ: %s vs %s", expected.versions, manifest.versions)
        if diffs:
            raise ReproducibilityError(f"rebuild differs from {paths.features_manifest}: {diffs}")
        log.info(
            "rebuild matches %s (features sha256 %s)",
            paths.features_manifest,
            manifest.features_sha256,
        )
        return manifest

    for path, frame in ((paths.canonical, canonical), (paths.features, features)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression="zstd", statistics=True)
        log.info("wrote %s (%d rows, %d cols)", path, frame.height, frame.width)
    write_json(manifest, paths.features_manifest)
    log.info("wrote %s", paths.features_manifest)
    return manifest
