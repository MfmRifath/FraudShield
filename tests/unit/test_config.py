from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from fraudshield.config import CONFIG_ENV_VAR, DayRange, Settings, SplitConfig, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_YAML = REPO_ROOT / "configs/base.yaml"
# Every runtime profile file must validate against Settings (ADR-0003). Add new profiles here.
PROFILES = ["configs/base.yaml"]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run each test from the repo root with no FRAUDSHIELD_* overrides leaking in."""
    for key in list(os.environ):
        if key.startswith("FRAUDSHIELD_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(REPO_ROOT)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _use_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]) -> None:
    """Point FRAUDSHIELD_CONFIG at base.yaml deep-merged with ``overrides``."""
    body = _merge(yaml.safe_load(BASE_YAML.read_text()), overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(body))
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))


@pytest.mark.unit
def test_base_yaml_loads() -> None:
    s = Settings()
    assert s.env == "dev"
    assert s.seed == 42
    assert s.paths.data_raw == Path("data/raw")
    assert s.dataset == "ieee_cis"


@pytest.mark.unit
@pytest.mark.parametrize("profile", PROFILES)
def test_profile_validates(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = REPO_ROOT / profile
    assert path.is_file()
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    Settings()


@pytest.mark.unit
def test_ieee_cis_defaults_match_spec() -> None:
    ds = Settings().datasets.ieee_cis
    assert ds.origin == datetime(2017, 12, 1, tzinfo=UTC)
    sp = ds.splits
    assert (sp.train.start, sp.train.end) == (1, 120)
    assert (sp.val.start, sp.val.end) == (128, 150)
    assert (sp.test.start, sp.test.end) == (158, 183)
    assert sp.gap_days == 7


@pytest.mark.unit
def test_config_file_selected_by_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_yaml(tmp_path, monkeypatch, {"env": "test", "seed": 7})
    s = Settings()
    assert (s.env, s.seed) == ("test", 7)


@pytest.mark.unit
def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "123")
    monkeypatch.setenv("FRAUDSHIELD_PATHS__DATA_RAW", "/mnt/raw")
    monkeypatch.setenv("FRAUDSHIELD_DATASET", "sparkov")
    s = Settings()
    assert s.seed == 123
    assert s.paths.data_raw == Path("/mnt/raw")
    assert s.active_dataset() == s.datasets.sparkov
    # sibling nested keys still come from YAML / defaults
    assert s.paths.reports == Path("reports")


@pytest.mark.unit
def test_init_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "123")
    assert Settings(seed=1).seed == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"sed": 1},
        {"paths": {"data_rwa": "x"}},
        {"datasets": {"sparkov": {"orign": "2019-01-01T00:00:00Z"}}},
    ],
    ids=["top-level", "nested", "dataset"],
)
def test_unknown_yaml_key_is_rejected(
    overrides: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_yaml(tmp_path, monkeypatch, overrides)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings()


@pytest.mark.unit
def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "-1")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.unit
def test_missing_config_file_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "does-not-exist.yaml"))
    with pytest.raises(ValidationError, match="datasets"):
        Settings()


@pytest.mark.unit
def test_naive_origin_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_yaml(tmp_path, monkeypatch, {"datasets": {"sparkov": {"origin": "2019-01-01T00:00:00"}}})
    with pytest.raises(ValidationError, match="timezone-aware"):
        Settings()


@pytest.mark.unit
def test_label_delay_longer_than_gap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_yaml(tmp_path, monkeypatch, {"labels": {"delay": {"max_days": 8.0}}})
    with pytest.raises(ValidationError, match="exceeds"):
        Settings()


@pytest.mark.unit
def test_split_gap_is_enforced() -> None:
    with pytest.raises(ValidationError, match="gap between train"):
        SplitConfig(
            train=DayRange(start=1, end=120),
            val=DayRange(start=127, end=150),  # only 6 gap days
            test=DayRange(start=158, end=183),
        )


@pytest.mark.unit
def test_day_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        DayRange(start=5, end=4)


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
