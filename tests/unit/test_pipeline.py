from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from fraudshield.config import PathsConfig, Settings
from fraudshield.manifest import FeaturesManifest, RawManifest, frame_content_sha256, read_json
from fraudshield.pipeline import (
    DatasetPaths,
    ReproducibilityError,
    build_features,
    fetch_raw,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sparkov_sample.csv"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose data dirs live in tmp_path, with the fixture as the 'raw' Sparkov train file."""
    paths = PathsConfig(
        data_raw=tmp_path / "raw",
        data_interim=tmp_path / "interim",
        data_processed=tmp_path / "processed",
        manifests=tmp_path / "manifests",
    )
    raw = paths.data_raw / "sparkov"
    raw.mkdir(parents=True)
    shutil.copy(FIXTURE, raw / "fraudTrain.csv")
    # header-only test file: the adapter concatenates both files
    (raw / "fraudTest.csv").write_text(FIXTURE.read_text().splitlines()[0] + "\n")
    return Settings(paths=paths, dataset="sparkov")


@pytest.mark.unit
def test_fetch_raw_records_then_verifies_manifest(settings: Settings) -> None:
    m1 = fetch_raw(settings, "sparkov")  # files present -> no download, manifest recorded
    paths = DatasetPaths.of(settings, "sparkov")
    assert read_json(RawManifest, paths.raw_manifest) == m1
    assert [(f.name, f.rows) for f in m1.files] == [("fraudTrain.csv", 2000), ("fraudTest.csv", 0)]
    assert fetch_raw(settings, "sparkov") == m1  # second call verifies against it

    with (paths.raw_dir / "fraudTrain.csv").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ReproducibilityError, match="differ"):
        fetch_raw(settings, "sparkov")


@pytest.mark.unit
def test_build_features_is_reproducible_and_checked(settings: Settings) -> None:
    fetch_raw(settings, "sparkov")
    m = build_features(settings, "sparkov")
    paths = DatasetPaths.of(settings, "sparkov")
    assert read_json(FeaturesManifest, paths.features_manifest) == m
    feats = pl.read_parquet(paths.features)
    assert feats.height == m.rows == 2000
    assert frame_content_sha256(feats) == m.features_sha256  # parquet round-trip is lossless
    assert frame_content_sha256(pl.read_parquet(paths.canonical)) == m.canonical_sha256

    assert build_features(settings, "sparkov", check=True) == m  # rebuild -> identical hashes

    # A changed feature definition (smoothing) must be caught by --check.
    changed = settings.model_copy(
        update={"features": settings.features.model_copy(update={"target_encoding_smoothing": 1.0})}
    )
    with pytest.raises(ReproducibilityError, match="feature_set_hash"):
        build_features(changed, "sparkov", check=True)


@pytest.mark.unit
def test_build_features_requires_raw_manifest(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        build_features(settings, "sparkov")
