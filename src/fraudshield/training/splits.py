"""Time-based dataset splits with label-delay gaps, and rolling-origin CV inside train.

Day numbers: ``day = floor((event_time - origin) / 1 day)``; for IEEE-CIS (origin 2017-12-01)
this equals ``floor(TransactionDT / 86400)``, so the first day is 1.

Gap days between train/val and val/test are dropped entirely: a model trained up to day ``d``
is only evaluated on days whose labels it could not have seen, and every training label has
matured (``labels.delay.max_days <= gap_days`` is validated in config).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import polars as pl

from fraudshield.config import DayRange, SplitConfig

SplitName = Literal["train", "val", "test"]
SPLITS: Final[tuple[SplitName, ...]] = ("train", "val", "test")
_US_PER_DAY: Final = 86_400 * 1_000_000


def day_number(origin: datetime, col: str = "event_time") -> pl.Expr:
    since = pl.col(col) - pl.lit(origin).dt.cast_time_unit("us")
    return (since.dt.total_microseconds() // _US_PER_DAY).cast(pl.Int32)


def _in(day: pl.Expr, r: DayRange) -> pl.Expr:
    return day.is_between(r.start, r.end, closed="both")


def assign_splits(df: pl.DataFrame, origin: datetime, cfg: SplitConfig) -> pl.DataFrame:
    """Add ``day`` (Int32) and ``split`` (train/val/test, null for gap or out-of-range days)."""
    day = day_number(origin)
    split = (
        pl.when(_in(day, cfg.train))
        .then(pl.lit("train"))
        .when(_in(day, cfg.val))
        .then(pl.lit("val"))
        .when(_in(day, cfg.test))
        .then(pl.lit("test"))
    )
    return df.with_columns(day=day, split=split)


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train: DayRange
    val: DayRange


def rolling_origin_folds(cfg: SplitConfig) -> list[Fold]:
    """Expanding-window folds inside ``cfg.train``, chronological, the last ending at train end.

    Fold k validates on ``cv_val_days`` consecutive days and trains on every train day up to
    ``gap_days`` before its validation window.
    """
    folds: list[Fold] = []
    for k in range(cfg.cv_folds):
        val_end = cfg.train.end - k * cfg.cv_val_days
        val_start = val_end - cfg.cv_val_days + 1
        train_end = val_start - cfg.gap_days - 1
        train = DayRange(start=cfg.train.start, end=max(train_end, cfg.train.start))
        if train_end - cfg.train.start + 1 < cfg.cv_min_train_days:
            raise ValueError(
                f"fold {k}: only {train_end - cfg.train.start + 1} training days "
                f"(< cv_min_train_days={cfg.cv_min_train_days}); reduce cv_folds or cv_val_days"
            )
        folds.append(Fold(index=0, train=train, val=DayRange(start=val_start, end=val_end)))
    folds.reverse()
    return [Fold(index=i, train=f.train, val=f.val) for i, f in enumerate(folds)]
