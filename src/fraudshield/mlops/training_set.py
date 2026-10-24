"""Build the training set as of a point in time from chargeback and analyst labels.

* Chargeback labels come from the dataset's ground truth and count once *matured*:
  ``label_available_time <= as_of`` (ADR-0005).
* Analyst labels come from review cases and count from their ``label_time <= as_of``; if a
  transaction was labelled several times, the latest label as of ``as_of`` is used.
* When both exist, **the chargeback wins** (it is the ground truth; analysts can be wrong).
* Only events with ``event_time <= as_of`` are eligible. Rows without any label known at
  ``as_of`` are dropped.

The output keeps every feature column. ``label`` / ``label_available_time`` are replaced by the
resolved label and its time, and ``label_source`` is added.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from fraudshield.cases.store import Case

ANALYST_SCHEMA: dict[str, pl.DataType] = {
    "txn_id": pl.String(),
    "label": pl.Int8(),
    "label_time": pl.Datetime("us", "UTC"),
    "analyst_id": pl.String(),
}


@dataclass(frozen=True)
class TrainingSetStats:
    as_of: datetime
    eligible_events: int
    chargeback_labels: int
    analyst_labels: int
    conflicts: int
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.__dict__.items()
        }


def analyst_labels_from_cases(cases: Iterable[Case]) -> pl.DataFrame:
    rows = [
        {
            "txn_id": c.case_id,
            "label": 1 if c.label == "fraud" else 0,
            "label_time": c.labeled_at,
            "analyst_id": c.analyst_id,
        }
        for c in cases
        if c.status == "labeled" and c.label is not None and c.labeled_at is not None
    ]
    return pl.DataFrame(rows, schema=ANALYST_SCHEMA)


def build_training_set(
    features: pl.DataFrame, analyst: pl.DataFrame | None, as_of: datetime
) -> tuple[pl.DataFrame, TrainingSetStats]:
    eligible = features.filter(pl.col("event_time") <= as_of)
    cb = eligible.filter(
        pl.col("label").is_not_null() & (pl.col("label_available_time") <= as_of)
    ).select(
        "txn_id", cb_label=pl.col("label").cast(pl.Int8), cb_time=pl.col("label_available_time")
    )
    an_src = analyst if analyst is not None else pl.DataFrame(schema=ANALYST_SCHEMA)
    an = (
        an_src.filter(pl.col("label_time") <= as_of)
        .sort("label_time")
        .group_by("txn_id", maintain_order=True)
        .last()
        .select("txn_id", an_label=pl.col("label").cast(pl.Int8), an_time=pl.col("label_time"))
    )
    resolved = (
        eligible.drop("label", "label_available_time")
        .join(cb, on="txn_id", how="left", maintain_order="left")
        .join(an, on="txn_id", how="left", maintain_order="left")
    )
    conflicts = resolved.filter(
        pl.col("cb_label").is_not_null()
        & pl.col("an_label").is_not_null()
        & (pl.col("cb_label") != pl.col("an_label"))
    ).height
    has_cb = pl.col("cb_label").is_not_null()
    out = (
        resolved.with_columns(
            label=pl.when(has_cb).then(pl.col("cb_label")).otherwise(pl.col("an_label")),
            label_available_time=pl.when(has_cb)
            .then(pl.col("cb_time"))
            .otherwise(pl.col("an_time")),
            label_source=pl.when(has_cb)
            .then(pl.lit("chargeback"))
            .when(pl.col("an_label").is_not_null())
            .then(pl.lit("analyst")),
        )
        .filter(pl.col("label").is_not_null())
        .drop("cb_label", "cb_time", "an_label", "an_time")
        .select([*features.columns, "label_source"])
    )
    stats = TrainingSetStats(
        as_of=as_of,
        eligible_events=eligible.height,
        chargeback_labels=cb.height,
        analyst_labels=an.join(eligible.select("txn_id"), on="txn_id", how="semi").height,
        conflicts=conflicts,
        rows=out.height,
    )
    return out, stats
