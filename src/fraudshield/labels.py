"""Label-availability model (ADR-0005).

Historical datasets only say *whether* a transaction was fraud, not *when* that became known.
We model the label delay as a deterministic function of ``(seed, txn_id)`` so that the offline
feature executor, the streaming label emitter and the training-set builder agree exactly.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Final

import polars as pl

from fraudshield.config import LabelDelayConfig

_US_PER_DAY: Final = 86_400 * 1_000_000
_U64: Final = float(2**64)


def unit_hash(txn_id: str, seed: int) -> float:
    """Stable pseudo-uniform value in [0, 1) for ``txn_id`` (independent of Python's hash seed)."""
    digest = hashlib.blake2b(f"{seed}:{txn_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _U64


def label_delay_us(txn_id: str, cfg: LabelDelayConfig) -> int:
    """Label delay in whole microseconds. Always >= 1 µs so a label is never known at event time."""
    days = cfg.min_days
    if cfg.kind == "uniform":
        days += (cfg.max_days - cfg.min_days) * unit_hash(txn_id, cfg.seed)
    return max(1, round(days * _US_PER_DAY))


def label_available_time(txn_id: str, event_time: datetime, cfg: LabelDelayConfig) -> datetime:
    return event_time + timedelta(microseconds=label_delay_us(txn_id, cfg))


def with_label_available_time(df: pl.DataFrame, cfg: LabelDelayConfig) -> pl.DataFrame:
    """Set ``label_available_time`` for rows with a label (null otherwise)."""
    delays = pl.Series(
        "_delay_us", [label_delay_us(t, cfg) for t in df["txn_id"].to_list()], dtype=pl.Int64
    )
    return (
        df.with_columns(delays)
        .with_columns(
            label_available_time=pl.when(pl.col("label").is_not_null()).then(
                pl.col("event_time") + pl.duration(microseconds=pl.col("_delay_us"))
            )
        )
        .drop("_delay_us")
    )
