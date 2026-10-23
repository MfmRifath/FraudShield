"""Typed application settings.

Precedence (highest first): explicit init kwargs > environment variables > YAML file > defaults.

* The YAML file defaults to ``configs/base.yaml`` and can be switched with ``FRAUDSHIELD_CONFIG``.
* Environment variables use the ``FRAUDSHIELD_`` prefix and ``__`` for nesting,
  e.g. ``FRAUDSHIELD_PATHS__DATA_RAW=/mnt/raw``.
* Unknown keys in the YAML file are rejected so typos fail loudly.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_ENV_VAR = "FRAUDSHIELD_CONFIG"
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

DatasetName = Literal["ieee_cis", "sparkov"]


def config_path() -> Path:
    """Return the YAML config path selected by ``FRAUDSHIELD_CONFIG`` (or the default)."""
    return Path(os.environ.get(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Frozen):
    """Filesystem layout. Relative paths are resolved against the working directory."""

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    manifests: Path = Path("data/manifests")
    reports: Path = Path("reports")
    fixtures: Path = Path("tests/fixtures")
    costs: Path = Path("configs/costs.yaml")
    rules: Path = Path("configs/rules.yaml")
    tuned_params: Path = Path("configs/tuned")
    model_card: Path = Path("docs/model_card.md")


class DayRange(_Frozen):
    """Inclusive range of dataset day numbers (day 1 = first day after the dataset origin)."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"day range end {self.end} < start {self.start}")
        return self

    @property
    def n_days(self) -> int:
        return self.end - self.start + 1


class SplitConfig(_Frozen):
    """Time-based train/val/test split with label-delay gaps, plus rolling-origin CV inside train."""

    train: DayRange
    val: DayRange
    test: DayRange
    gap_days: int = Field(default=7, ge=0)
    cv_folds: int = Field(default=3, ge=1)
    cv_val_days: int = Field(default=14, ge=1)
    cv_min_train_days: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def _gaps(self) -> Self:
        for a_name, a, b_name, b in (
            ("train", self.train, "val", self.val),
            ("val", self.val, "test", self.test),
        ):
            gap = b.start - a.end - 1
            if gap < self.gap_days:
                raise ValueError(
                    f"gap between {a_name} (ends {a.end}) and {b_name} (starts {b.start}) "
                    f"is {gap} days; need >= {self.gap_days}"
                )
        return self


class DatasetConfig(_Frozen):
    """Per-dataset settings."""

    kaggle_handle: str
    kaggle_kind: Literal["dataset", "competition"]
    license: str
    origin: datetime = Field(
        description="Day numbering origin (UTC); day = floor((t - origin) / 1d)"
    )
    splits: SplitConfig
    extra_columns: tuple[str, ...] = ()

    @field_validator("origin")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("origin must be timezone-aware (e.g. 2017-12-01T00:00:00Z)")
        return v.astimezone(UTC)


class DatasetsConfig(_Frozen):
    ieee_cis: DatasetConfig
    sparkov: DatasetConfig

    def get(self, name: DatasetName) -> DatasetConfig:
        cfg: DatasetConfig = getattr(self, name)
        return cfg


class LabelDelayConfig(_Frozen):
    """Event-time delay between a transaction and the moment its label becomes known.

    The delay is a deterministic function of (seed, txn_id) so offline features, the streaming
    label emitter and tests all agree on when each label becomes available.
    """

    kind: Literal["fixed", "uniform"] = "uniform"
    min_days: float = Field(default=1.0, ge=0)
    max_days: float = Field(default=7.0, ge=0)
    seed: int = 17

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.max_days < self.min_days:
            raise ValueError("label delay max_days < min_days")
        return self


class LabelsConfig(_Frozen):
    delay: LabelDelayConfig = LabelDelayConfig()


class FeaturesConfig(_Frozen):
    """Tunables for the feature library (definitions live in code: features/definitions.py)."""

    target_encoding_smoothing: float = Field(default=50.0, gt=0)


class LgbmConfig(_Frozen):
    """LightGBM defaults (overridden by ``configs/tuned/<dataset>.json`` when present)."""

    num_leaves: int = Field(default=63, ge=2)
    learning_rate: float = Field(default=0.05, gt=0)
    num_boost_round: int = Field(default=2000, ge=1)
    early_stopping_rounds: int = Field(default=100, ge=1)
    min_child_samples: int = Field(default=100, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l2: float = Field(default=1.0, ge=0)
    scale_pos_weight: float = Field(default=1.0, gt=0)
    num_threads: int = Field(default=0, ge=0, description="training threads; 0 = LightGBM default")


class TuneConfig(_Frozen):
    n_trials: int = Field(default=40, ge=1)
    timeout_s: int | None = Field(default=None, ge=1)


class TrainingConfig(_Frozen):
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_location: str = "mlartifacts"
    experiment: str = "fraudshield"
    registered_model: str = "fraudshield"
    categorical_min_count: int = Field(default=20, ge=1)
    platt_below_positives: int = Field(default=1000, ge=0)
    lgbm: LgbmConfig = LgbmConfig()
    tune: TuneConfig = TuneConfig()


class Settings(BaseSettings):
    """Root settings object. Later phases add sections (serving, streaming, ...)."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUDSHIELD_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    env: Literal["dev", "test", "ci", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    seed: int = Field(default=42, ge=0)
    paths: PathsConfig = PathsConfig()
    dataset: DatasetName = "ieee_cis"
    datasets: DatasetsConfig
    labels: LabelsConfig = LabelsConfig()
    features: FeaturesConfig = FeaturesConfig()
    training: TrainingConfig = TrainingConfig()

    @model_validator(mode="after")
    def _labels_mature_within_gap(self) -> Self:
        # Labels of the last training day must be available before validation starts.
        max_delay = self.labels.delay.max_days
        for name, ds in (("ieee_cis", self.datasets.ieee_cis), ("sparkov", self.datasets.sparkov)):
            gap = ds.splits.gap_days
            if max_delay > gap:
                raise ValueError(
                    f"labels.delay.max_days={max_delay} exceeds {name} split gap_days={gap}"
                )
        return self

    def active_dataset(self) -> DatasetConfig:
        return self.datasets.get(self.dataset)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return (init_settings, env_settings, yaml_settings, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call ``get_settings.cache_clear()`` in tests after env changes."""
    return Settings()
