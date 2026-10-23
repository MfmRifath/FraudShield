"""Typed application settings.

Precedence (highest first): explicit init kwargs > environment variables > YAML file > defaults.

* The YAML file defaults to ``configs/base.yaml`` and can be switched with ``FRAUDSHIELD_CONFIG``.
* Environment variables use the ``FRAUDSHIELD_`` prefix and ``__`` for nesting,
  e.g. ``FRAUDSHIELD_PATHS__DATA_RAW=/mnt/raw``.
* Unknown keys in the YAML file are rejected so typos fail loudly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_ENV_VAR = "FRAUDSHIELD_CONFIG"
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def config_path() -> Path:
    """Return the YAML config path selected by ``FRAUDSHIELD_CONFIG`` (or the default)."""
    return Path(os.environ.get(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths are resolved against the working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    reports: Path = Path("reports")
    fixtures: Path = Path("tests/fixtures")


class Settings(BaseSettings):
    """Root settings object. Later phases add sections (features, training, serving, ...)."""

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
