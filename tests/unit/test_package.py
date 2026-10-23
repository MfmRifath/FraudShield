from __future__ import annotations

import importlib

import pytest

import fraudshield

SUBPACKAGES = [
    "adapters",
    "features",
    "streaming",
    "training",
    "serving",
    "monitoring",
    "flows",
    "cases",
]


@pytest.mark.unit
def test_version_is_set() -> None:
    assert fraudshield.__version__ not in ("", "0.0.0")


@pytest.mark.unit
@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    module = importlib.import_module(f"fraudshield.{name}")
    assert module.__doc__
