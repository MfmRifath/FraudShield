from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from fraudshield.config import CONFIG_ENV_VAR, Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return path


@pytest.mark.unit
def test_base_yaml_loads() -> None:
    s = Settings()
    assert s.env == "dev"
    assert s.seed == 42
    assert s.paths.data_raw == Path("data/raw")


@pytest.mark.unit
@pytest.mark.parametrize("profile", PROFILES)
def test_profile_validates(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = REPO_ROOT / profile
    assert path.is_file()
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    Settings()


@pytest.mark.unit
def test_config_file_selected_by_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(_write_yaml(tmp_path, "env: test\nseed: 7\n")))
    s = Settings()
    assert (s.env, s.seed) == ("test", 7)


@pytest.mark.unit
def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "123")
    monkeypatch.setenv("FRAUDSHIELD_PATHS__DATA_RAW", "/mnt/raw")
    s = Settings()
    assert s.seed == 123
    assert s.paths.data_raw == Path("/mnt/raw")
    # sibling nested keys still come from YAML / defaults
    assert s.paths.reports == Path("reports")


@pytest.mark.unit
def test_init_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "123")
    assert Settings(seed=1).seed == 1


@pytest.mark.unit
def test_unknown_yaml_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(_write_yaml(tmp_path, "sed: 1\n")))
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.unit
def test_unknown_nested_yaml_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(_write_yaml(tmp_path, "paths:\n  data_rwa: x\n")))
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.unit
def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUDSHIELD_SEED", "-1")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.unit
def test_missing_config_file_falls_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "does-not-exist.yaml"))
    assert Settings() == Settings.model_construct()


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
