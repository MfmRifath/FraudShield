"""Shared helpers for serving tests: synthetic settings, bundles and request payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pydantic import SecretStr

from fraudshield.config import PathsConfig, Settings
from fraudshield.synthetic import (
    ORIGIN,
    SPLITS,
    build_synthetic_bundle,
    synthetic_canonical,
    synthetic_settings,
)
from fraudshield.training.splits import assign_splits

REPO = Path(__file__).resolve().parents[1]
TOKEN = "test-admin-token"  # noqa: S105 - test fixture value


def serving_settings(
    tmp: Path, *, rules: Path | None = None, costs: Path | None = None, **serving: Any
) -> Settings:
    base = synthetic_settings(Settings())
    paths = PathsConfig(
        costs=costs or REPO / "configs/costs.yaml", rules=rules or REPO / "configs/rules.yaml"
    )
    srv = base.serving.model_copy(
        update={"dataset": "sparkov", "bundle_dir": tmp / "bundle", "admin_token": SecretStr(TOKEN)}
        | serving
    )
    return base.model_copy(update={"paths": paths, "serving": srv})


def make_bundle(settings: Settings, out: Path, rounds: int = 20) -> Path:
    path: Path = build_synthetic_bundle(settings, out, rounds=rounds)
    return path


def canonical_rows(n: int = 10_000) -> list[dict[str, Any]]:
    return synthetic_canonical(n, seed=0).to_dicts()


def window_rows() -> list[dict[str, Any]]:
    df = assign_splits(synthetic_canonical(10_000, seed=0), ORIGIN, SPLITS)
    return df.filter(pl.col("split") == "test").drop("day", "split").to_dicts()


def payload(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical row -> /score JSON body."""
    keys = ("txn_id", "uid", "amount", "product", "card_network", "card_kind", "p_email_domain",
            "r_email_domain", "device_type", "device_info", "addr")  # fmt: skip
    body = {k: row.get(k) for k in keys}
    body["event_time"] = row["event_time"].isoformat()
    body["extra"] = {k.removeprefix("x_").upper(): v for k, v in row.items() if k.startswith("x_")}
    return body


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


ALWAYS_REVIEW_RULES = """\
rules:
  - name: review_everything
    action: review
    reason: "test rule: every transaction goes to review"
    all:
      - {feature: amount, op: ge, value: 0}
"""
