"""Adapter base class and the shared canonicalisation step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

import polars as pl

from fraudshield.config import DatasetConfig, DatasetName, LabelDelayConfig
from fraudshield.labels import with_label_available_time
from fraudshield.schema import CANONICAL_COLUMNS, EXTRA_PREFIX, validate_canonical

# Tie-break offsets are added in microseconds; source data has whole-second resolution.
_MAX_TIES_PER_INSTANT: Final = 1_000_000


def extra_column_name(raw_name: str) -> str:
    return EXTRA_PREFIX + raw_name.lower()


def make_event_times_unique(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (event_time, txn_id) and add k µs to the k-th event sharing an instant (ADR-0004).

    The result has strictly increasing ``event_time``, which is the total order every
    point-in-time feature relies on.
    """
    df = df.sort(["event_time", "txn_id"])
    offset = pl.int_range(pl.len(), dtype=pl.Int64).over("event_time")
    max_offset = df.select(offset.max()).item()
    if max_offset is not None and max_offset >= _MAX_TIES_PER_INSTANT:
        raise ValueError(f"{max_offset + 1} events share one instant; cannot tie-break in µs")
    out = df.with_columns(pl.col("event_time") + pl.duration(microseconds=offset))
    if not out["event_time"].is_unique().all() or not out["event_time"].is_sorted():
        raise ValueError("tie-break produced colliding event times (sub-second source data?)")
    return out


def finalize_canonical(
    df: pl.DataFrame, extra_names: Sequence[str], delay: LabelDelayConfig
) -> pl.DataFrame:
    """Shared tail of every adapter: fill/cast columns, unique times, label times, order, validate."""
    missing = {
        c: pl.lit(None, dtype=t)
        for c, t in CANONICAL_COLUMNS.items()
        if c not in df.columns and c != "label_available_time"
    }
    df = df.with_columns(**missing).with_columns(
        [pl.col(c).cast(t) for c, t in CANONICAL_COLUMNS.items() if c != "label_available_time"]
    )
    df = make_event_times_unique(df)
    df = with_label_available_time(df, delay)
    extras = [extra_column_name(n) for n in extra_names]
    df = df.select([*CANONICAL_COLUMNS, *(pl.col(c).cast(pl.Float64) for c in extras)])
    return validate_canonical(df)


class DatasetAdapter(ABC):
    """Maps one raw dataset to the canonical transaction frame."""

    name: ClassVar[DatasetName]
    raw_files: ClassVar[tuple[str, ...]]

    def __init__(self, cfg: DatasetConfig, delay: LabelDelayConfig) -> None:
        self.cfg = cfg
        self.delay = delay

    def default_files(self, raw_dir: Path) -> list[Path]:
        return [raw_dir / f for f in self.raw_files]

    @abstractmethod
    def read_raw(self, files: Sequence[Path]) -> pl.DataFrame:
        """Read source files into a frame with canonical column names (any order, unsorted)."""

    def load(self, files: Sequence[Path]) -> pl.DataFrame:
        for f in files:
            if not f.is_file():
                raise FileNotFoundError(f"{self.name}: missing raw file {f} (run `make data`)")
        return finalize_canonical(self.read_raw(files), self.cfg.extra_columns, self.delay)
