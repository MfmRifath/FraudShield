from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fraudshield.adapters import get_adapter
from fraudshield.config import Settings
from fraudshield.eda import (
    amount_summary,
    daily_class_balance,
    new_uid_share_by_day,
    new_vs_known_summary,
    uid_summary,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sparkov_sample.csv"


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return get_adapter("sparkov", Settings()).load([FIXTURE])


@pytest.mark.unit
def test_daily_class_balance_sums_to_totals(df: pl.DataFrame) -> None:
    origin = Settings().datasets.sparkov.origin
    out = daily_class_balance(df, origin)
    assert out["n"].sum() == df.height
    assert out["n_fraud"].sum() == df["label"].sum()
    assert out["day"].is_sorted()
    assert out["day"].min() == 60  # fixture window starts 2019-03-01 = day 60 from 2018-12-31


@pytest.mark.unit
def test_amount_summary_by_label(df: pl.DataFrame) -> None:
    out = amount_summary(df)
    assert out["label"].to_list() == [0, 1]
    assert out["n"].sum() == df.height
    assert (out["p50"] <= out["p90"]).all()
    assert (out["p90"] <= out["max"]).all()


@pytest.mark.unit
def test_uid_and_new_uid_summaries(df: pl.DataFrame) -> None:
    u = uid_summary(df).row(0, named=True)
    assert u["n_uids"] == df["uid"].n_unique()
    assert 0 < u["share_uids_with_fraud"] < 1
    new = new_uid_share_by_day(df, Settings().datasets.sparkov.origin)
    total_new = (new["new_share"] * new["n"]).sum()
    assert round(total_new) == df["uid"].n_unique()  # each uid is "new" exactly once
    pooled = {r["segment"]: r for r in new_vs_known_summary(df).to_dicts()}
    assert pooled["new"]["n"] == df["uid"].n_unique()
    assert pooled["new"]["n"] + pooled["known"]["n"] == df.height
    assert pooled["new"]["n_fraud"] + pooled["known"]["n_fraud"] == df["label"].sum()
