"""Canonical transaction schema shared by every adapter, feature executor and service.

Two representations of the same thing:

* ``Transaction``: a pydantic record for single events (API payloads, streaming messages).
* A canonical Polars frame with the fixed columns in ``CANONICAL_COLUMNS`` plus any number of
  ``x_*`` Float64 columns holding the ``extra`` numeric fields. The frame is validated with
  pandera (``validate_canonical``).

Invariants of a canonical frame (enforced by ``validate_canonical``):

* sorted by ``event_time`` and ``event_time`` is **unique**: the strict total order that defines
  "strictly before" for point-in-time features (ADR-0004);
* ``txn_id`` unique; ``amount >= 0``; ``label`` in {0, 1} or null;
* ``label_available_time`` is set iff ``label`` is set, and is strictly after ``event_time``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Final, Literal

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["approve", "review", "block"]
DECISIONS: Final[tuple[Decision, ...]] = ("approve", "review", "block")

EXTRA_PREFIX: Final = "x_"
TIME_DTYPE: Final = pl.Datetime("us", "UTC")

CANONICAL_COLUMNS: Final[dict[str, pl.DataType]] = {
    "txn_id": pl.String(),
    "event_time": TIME_DTYPE,
    "uid": pl.String(),
    "amount": pl.Float64(),
    "product": pl.String(),
    "card_network": pl.String(),
    "card_kind": pl.String(),
    "p_email_domain": pl.String(),
    "r_email_domain": pl.String(),
    "device_type": pl.String(),
    "device_info": pl.String(),
    "addr": pl.String(),
    "label": pl.Int8(),
    "label_available_time": TIME_DTYPE,
}

_NULLABLE: Final = frozenset(CANONICAL_COLUMNS) - {"txn_id", "event_time", "uid", "amount"}


class Transaction(BaseModel):
    """One canonical transaction event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    txn_id: str = Field(min_length=1)
    event_time: datetime
    uid: str = Field(min_length=1)
    amount: float = Field(ge=0)
    product: str | None = None
    card_network: str | None = None
    card_kind: str | None = None
    p_email_domain: str | None = None
    r_email_domain: str | None = None
    device_type: str | None = None
    device_info: str | None = None
    addr: str | None = None
    extra: dict[str, float | None] = Field(default_factory=dict)
    label: int | None = Field(default=None, ge=0, le=1)
    label_available_time: datetime | None = None

    @field_validator("event_time", "label_available_time")
    @classmethod
    def _aware_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return v.astimezone(UTC)


def extra_columns(df: pl.DataFrame) -> list[str]:
    """The ``x_*`` columns of a canonical frame, in frame order."""
    return [c for c in df.columns if c.startswith(EXTRA_PREFIX)]


def to_records(df: pl.DataFrame) -> list[Transaction]:
    """Canonical frame -> list of ``Transaction``."""
    extras = extra_columns(df)
    out: list[Transaction] = []
    for row in df.iter_rows(named=True):
        fixed = {k: row[k] for k in CANONICAL_COLUMNS}
        extra = {c.removeprefix(EXTRA_PREFIX): row[c] for c in extras}
        out.append(Transaction(**fixed, extra=extra))
    return out


def from_records(records: Iterable[Transaction | Mapping[str, Any]]) -> pl.DataFrame:
    """List of ``Transaction`` (or dicts) -> canonical frame (column order + dtypes, unsorted)."""
    rows: list[dict[str, Any]] = []
    extra_keys: dict[str, None] = {}
    for rec in records:
        txn = rec if isinstance(rec, Transaction) else Transaction.model_validate(rec)
        d = txn.model_dump(exclude={"extra"})
        for k, v in txn.extra.items():
            d[EXTRA_PREFIX + k] = v
            extra_keys[EXTRA_PREFIX + k] = None
        rows.append(d)
    schema: dict[str, pl.DataType] = dict(CANONICAL_COLUMNS)
    schema.update({k: pl.Float64() for k in extra_keys})
    return pl.DataFrame(rows, schema=schema)


def _unique(col: str) -> pa.Check:
    return pa.Check(
        lambda data: data.lazyframe.select(pl.col(col).is_unique().all()), name=f"{col}_unique"
    )


def _sorted_strict(data: pa.PolarsData) -> pl.LazyFrame:
    t = pl.col("event_time")
    return data.lazyframe.select(
        ((t.diff() > pl.duration(microseconds=0)) | t.diff().is_null()).all()
    )


def _label_time_consistent(data: pa.PolarsData) -> pl.LazyFrame:
    has_label = pl.col("label").is_not_null()
    has_time = pl.col("label_available_time").is_not_null()
    after = pl.col("label_available_time") > pl.col("event_time")
    return data.lazyframe.select(((has_label == has_time) & (~has_label | after)).all())


CANONICAL_SCHEMA: Final = pa.DataFrameSchema(
    columns={
        **{
            name: pa.Column(dtype, nullable=name in _NULLABLE, coerce=False)
            for name, dtype in CANONICAL_COLUMNS.items()
        },
        "amount": pa.Column(pl.Float64, pa.Check.ge(0), nullable=False),
        "label": pa.Column(pl.Int8, pa.Check.isin([0, 1]), nullable=True),
        rf"^{EXTRA_PREFIX}.+$": pa.Column(pl.Float64, nullable=True, regex=True, required=False),
    },
    checks=[
        _unique("txn_id"),
        pa.Check(_sorted_strict, name="event_time_strictly_increasing"),
        pa.Check(_label_time_consistent, name="label_available_time_consistent"),
    ],
    strict=True,
    name="canonical_transactions",
)


def validate_canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a canonical frame (raises ``pandera.errors.SchemaError`` on violation)."""
    ordered = [*CANONICAL_COLUMNS, *extra_columns(df)]
    if df.columns != ordered:
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        raise ValueError(f"canonical column order mismatch; missing={sorted(missing)}")
    return CANONICAL_SCHEMA.validate(df)
