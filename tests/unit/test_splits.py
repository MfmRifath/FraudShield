from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fraudshield.config import DayRange, Settings, SplitConfig
from fraudshield.training.splits import assign_splits, day_number, rolling_origin_folds

ORIGIN = datetime(2017, 12, 1, tzinfo=UTC)


@st.composite
def split_configs(draw: st.DrawFn) -> SplitConfig:
    gap = draw(st.integers(0, 10))
    t0 = draw(st.integers(0, 5))
    t1 = t0 + draw(st.integers(20, 200))
    v0 = t1 + gap + 1 + draw(st.integers(0, 3))
    v1 = v0 + draw(st.integers(0, 30))
    s0 = v1 + gap + 1 + draw(st.integers(0, 3))
    s1 = s0 + draw(st.integers(0, 30))
    return SplitConfig(
        train=DayRange(start=t0, end=t1),
        val=DayRange(start=v0, end=v1),
        test=DayRange(start=s0, end=s1),
        gap_days=gap,
        cv_folds=draw(st.integers(1, 4)),
        cv_val_days=draw(st.integers(1, 20)),
        cv_min_train_days=draw(st.integers(1, 10)),
    )


@pytest.mark.unit
def test_day_number_matches_transaction_dt_convention() -> None:
    dts = [86_400, 86_400 * 120 + 5, 86_400 * 183 - 1]
    df = pl.DataFrame({"event_time": [ORIGIN + timedelta(seconds=s) for s in dts]}).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC"))
    )
    assert df.select(day_number(ORIGIN))["event_time"].to_list() == [s // 86_400 for s in dts]


@pytest.mark.unit
@given(cfg=split_configs(), days=st.lists(st.integers(0, 400), min_size=1, max_size=200))
def test_assign_splits_respects_ranges_and_gaps(cfg: SplitConfig, days: list[int]) -> None:
    times = [ORIGIN + timedelta(days=d, seconds=17) for d in days]
    df = pl.DataFrame({"event_time": times}).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC"))
    )
    out = assign_splits(df, ORIGIN, cfg)
    for d, split in out.select("day", "split").iter_rows():
        for name, r in (("train", cfg.train), ("val", cfg.val), ("test", cfg.test)):
            assert (split == name) == (r.start <= d <= r.end)
    by = {
        s: out.filter(pl.col("split") == s)["day"]
        for s in ("train", "val", "test")
        if (out["split"] == s).any()
    }
    if "train" in by and "val" in by:
        assert by["val"].min() - by["train"].max() > cfg.gap_days  # type: ignore[operator]
    if "val" in by and "test" in by:
        assert by["test"].min() - by["val"].max() > cfg.gap_days  # type: ignore[operator]


@pytest.mark.unit
@given(cfg=split_configs())
def test_rolling_origin_folds_properties(cfg: SplitConfig) -> None:
    earliest_train_end = cfg.train.end - cfg.cv_folds * cfg.cv_val_days - cfg.gap_days
    if earliest_train_end - cfg.train.start + 1 < cfg.cv_min_train_days:
        with pytest.raises(ValueError, match="cv_min_train_days"):
            rolling_origin_folds(cfg)
        return
    folds = rolling_origin_folds(cfg)
    assert len(folds) == cfg.cv_folds
    assert [f.index for f in folds] == list(range(cfg.cv_folds))
    assert folds[-1].val.end == cfg.train.end
    for f in folds:
        assert f.train.start == cfg.train.start
        assert f.val.start - f.train.end - 1 == cfg.gap_days  # exact gap
        assert f.train.n_days >= cfg.cv_min_train_days
        assert f.val.n_days == cfg.cv_val_days
        assert cfg.train.start <= f.val.start
        assert f.val.end <= cfg.train.end
    for a, b in itertools.pairwise(folds):
        assert b.val.start == a.val.end + 1  # contiguous, non-overlapping validation windows
        assert b.train.end > a.train.end  # expanding


@pytest.mark.unit
def test_rolling_origin_folds_rejects_too_little_history() -> None:
    cfg = SplitConfig(
        train=DayRange(start=1, end=40),
        val=DayRange(start=48, end=50),
        test=DayRange(start=58, end=60),
        cv_folds=3,
        cv_val_days=10,
        cv_min_train_days=30,
    )
    with pytest.raises(ValueError, match="cv_min_train_days"):
        rolling_origin_folds(cfg)


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["ieee_cis", "sparkov"])
def test_configured_folds_are_valid(dataset: str) -> None:
    s = Settings()
    cfg = s.datasets.ieee_cis if dataset == "ieee_cis" else s.datasets.sparkov
    folds = rolling_origin_folds(cfg.splits)
    assert len(folds) == cfg.splits.cv_folds


@pytest.mark.unit
def test_ieee_default_folds_are_as_documented() -> None:
    folds = rolling_origin_folds(Settings().datasets.ieee_cis.splits)
    got = [((f.train.start, f.train.end), (f.val.start, f.val.end)) for f in folds]
    assert got == [((1, 71), (79, 92)), ((1, 85), (93, 106)), ((1, 99), (107, 120))]
