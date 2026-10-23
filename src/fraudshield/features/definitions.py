"""The single source of truth for every model feature.

Features are *declarative specs*. Each executor interprets the same specs:

* ``features.offline``: vectorised over a canonical frame (Polars/DuckDB), for training/backfill.
* ``features.online`` (Phase 3/4): per event, from per-entity state in a ``FeatureStore``.

Point-in-time semantics (shared by both executors, ADR-0004/0005). For event *i* at time *t*:

* Events are totally ordered by ``event_time`` (unique per dataset). "History" is every event
  strictly before *i* in that order **plus event i itself**.
* ``WindowAgg`` / ``DistinctCount`` use the half-open window ``(t - window, t]`` over the history
  of events with the same non-null entity value. Distinct counts ignore null values.
* ``FirstSeen`` is 1.0 if no strictly-earlier event has the same (entity, value) pair, else 0.0.
* ``TimeSinceLast`` is seconds since the previous event of the same entity (null if none).
* ``TargetEncoding`` is computed from a daily batch snapshot taken at ``floor_day(t)`` (00:00 UTC)
  using only labels with ``label_available_time <= snapshot``:
  ``(pos_c + m * p0) / (n_c + m)`` with ``p0 = pos_all / n_all`` (null if no labels yet).
* Any entity-keyed feature is null when the event's entity value is null.

Request features (``RequestFeature`` subclasses) depend only on the current event and expose both
a Polars expression and a scalar implementation, so offline and online share the definition.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Any, Literal

import polars as pl

from fraudshield.schema import EXTRA_PREFIX

Family = Literal[
    "request",
    "velocity",
    "amount",
    "ratio",
    "recency",
    "distinct",
    "first_seen",
    "target_encoding",
]
Agg = Literal["count", "sum", "mean", "max"]
Row = Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class Feature(ABC):
    name: str
    family: Family

    @property
    def categorical(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable spec, used for the feature-set hash."""
        out: dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.total_seconds() if isinstance(v, timedelta) else v
        return out


# ------------------------------------------------------------------ request-only features
@dataclass(frozen=True, kw_only=True)
class RequestFeature(Feature):
    family: Family = "request"

    @abstractmethod
    def expr(self) -> pl.Expr:
        """Vectorised implementation over a canonical frame."""

    @abstractmethod
    def scalar(self, row: Row) -> Any:
        """Single-event implementation over a canonical row mapping."""


@dataclass(frozen=True, kw_only=True)
class Passthrough(RequestFeature):
    column: str
    is_categorical: bool = False

    @property
    def categorical(self) -> bool:
        return self.is_categorical

    def expr(self) -> pl.Expr:
        e = pl.col(self.column)
        return (e if self.is_categorical else e.cast(pl.Float64)).alias(self.name)

    def scalar(self, row: Row) -> Any:
        v = row.get(self.column)
        if v is None or self.is_categorical:
            return v
        return float(v)


@dataclass(frozen=True, kw_only=True)
class Log1p(RequestFeature):
    column: str

    def expr(self) -> pl.Expr:
        return pl.col(self.column).cast(pl.Float64).log1p().alias(self.name)

    def scalar(self, row: Row) -> float | None:
        v = row.get(self.column)
        return None if v is None else math.log1p(float(v))


@dataclass(frozen=True, kw_only=True)
class Cents(RequestFeature):
    """Cents part of an amount as an integer 0..99 (``floor(x*100 + 0.5) mod 100``)."""

    column: str

    def expr(self) -> pl.Expr:
        c = (pl.col(self.column).cast(pl.Float64) * 100 + 0.5).floor()
        return (c % 100).alias(self.name)

    def scalar(self, row: Row) -> float | None:
        v = row.get(self.column)
        return None if v is None else float(math.floor(float(v) * 100 + 0.5) % 100)


@dataclass(frozen=True, kw_only=True)
class HourOfDay(RequestFeature):
    """UTC hour 0..23 of ``event_time``."""

    def expr(self) -> pl.Expr:
        return pl.col("event_time").dt.hour().cast(pl.Float64).alias(self.name)

    def scalar(self, row: Row) -> float:
        t: datetime = row["event_time"]
        return float(t.hour)


@dataclass(frozen=True, kw_only=True)
class DayOfWeek(RequestFeature):
    """ISO weekday 1 (Mon) .. 7 (Sun) of ``event_time`` in UTC."""

    def expr(self) -> pl.Expr:
        return pl.col("event_time").dt.weekday().cast(pl.Float64).alias(self.name)

    def scalar(self, row: Row) -> float:
        t: datetime = row["event_time"]
        return float(t.isoweekday())


@dataclass(frozen=True, kw_only=True)
class SameValue(RequestFeature):
    """1.0 if both columns are non-null and equal, 0.0 if both non-null and different, else null."""

    left: str
    right: str

    def expr(self) -> pl.Expr:
        a, b = pl.col(self.left), pl.col(self.right)
        return (
            pl.when(a.is_not_null() & b.is_not_null())
            .then((a == b).cast(pl.Float64))
            .alias(self.name)
        )

    def scalar(self, row: Row) -> float | None:
        a, b = row.get(self.left), row.get(self.right)
        return None if a is None or b is None else float(a == b)


# ------------------------------------------------------------------ history features
@dataclass(frozen=True, kw_only=True)
class WindowAgg(Feature):
    """Aggregate of ``value`` (or event count) over ``(t - window, t]`` for the event's entity."""

    entity: str
    agg: Agg
    window: timedelta
    value: str | None = None  # None only for agg == "count"

    def __post_init__(self) -> None:
        if (self.agg == "count") != (self.value is None):
            raise ValueError(f"{self.name}: value must be None iff agg == 'count'")


@dataclass(frozen=True, kw_only=True)
class DistinctCount(Feature):
    """Number of distinct non-null ``value`` over ``(t - window, t]`` for the event's entity."""

    family: Family = "distinct"
    entity: str
    value: str
    window: timedelta


@dataclass(frozen=True, kw_only=True)
class FirstSeen(Feature):
    """1.0 if the (entity, value) pair never occurred strictly before; null if value is null."""

    family: Family = "first_seen"
    entity: str
    value: str


@dataclass(frozen=True, kw_only=True)
class TimeSinceLast(Feature):
    """Seconds since the previous event with the same entity (null for the first)."""

    family: Family = "recency"
    entity: str


@dataclass(frozen=True, kw_only=True)
class Ratio(Feature):
    """``numerator / denominator`` of two other features (null if denominator is null or 0)."""

    family: Family = "ratio"
    numerator: str
    denominator: str


@dataclass(frozen=True, kw_only=True)
class TargetEncoding(Feature):
    """Smoothed fraud rate of ``column`` from the daily label snapshot (see module docstring)."""

    family: Family = "target_encoding"
    column: str


# ------------------------------------------------------------------ the feature set
@dataclass(frozen=True)
class FeatureSet:
    features: tuple[Feature, ...]
    te_smoothing: float

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate feature names: {dupes}")
        known = set(names)
        for f in self.features:
            if isinstance(f, Ratio):
                for dep in (f.numerator, f.denominator):
                    if dep not in known:
                        raise ValueError(f"{f.name}: unknown dependency {dep}")

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def categorical(self) -> list[str]:
        return [f.name for f in self.features if f.categorical]

    def of_type[T: Feature](self, cls: type[T]) -> list[T]:
        return [f for f in self.features if isinstance(f, cls)]

    def by_family(self, family: Family) -> list[Feature]:
        return [f for f in self.features if f.family == family]

    @property
    def request_features(self) -> list[RequestFeature]:
        return [f for f in self.features if isinstance(f, RequestFeature)]

    @property
    def request_only(self) -> list[str]:
        """Features usable without the feature store (the "degraded" model's inputs)."""
        return [f.name for f in self.by_family("request")]

    def spec_hash(self) -> str:
        payload = {
            "te_smoothing": self.te_smoothing,
            "features": [f.describe() for f in self.features],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


_W = {
    "10m": timedelta(minutes=10),
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

# Entity / value columns of the canonical schema used by history features.
UID, DEVICE, EMAIL, ADDR = "uid", "device_info", "p_email_domain", "addr"


def build_feature_set(extra_columns: Sequence[str], te_smoothing: float) -> FeatureSet:
    """Build the canonical feature set. ``extra_columns`` are raw dataset names (e.g. ``C1``)."""
    request: list[Feature] = [
        Passthrough(name="amount", column="amount"),
        Log1p(name="log_amount", column="amount"),
        Cents(name="amount_cents", column="amount"),
        Passthrough(name="product", column="product", is_categorical=True),
        Passthrough(name="card_network", column="card_network", is_categorical=True),
        Passthrough(name="card_kind", column="card_kind", is_categorical=True),
        Passthrough(name="p_email_domain", column="p_email_domain", is_categorical=True),
        Passthrough(name="r_email_domain", column="r_email_domain", is_categorical=True),
        SameValue(name="email_p_eq_r", left="p_email_domain", right="r_email_domain"),
        Passthrough(name="device_type", column="device_type", is_categorical=True),
        HourOfDay(name="hour"),
        DayOfWeek(name="weekday"),
        *(
            Passthrough(name=EXTRA_PREFIX + c.lower(), column=EXTRA_PREFIX + c.lower())
            for c in extra_columns
        ),
    ]
    velocity: list[Feature] = [
        WindowAgg(name=f"uid_cnt_{w}", family="velocity", entity=UID, agg="count", window=_W[w])
        for w in ("10m", "1h", "24h", "7d")
    ]
    amount: list[Feature] = [
        WindowAgg(
            name=f"uid_amt_{agg}_{w}",
            family="amount",
            entity=UID,
            agg=agg,
            window=_W[w],
            value="amount",
        )
        for agg in ("sum", "mean", "max")
        for w in ("1h", "24h", "7d")
    ]
    derived: list[Feature] = [
        Ratio(name="amt_to_uid_mean_7d", numerator="amount", denominator="uid_amt_mean_7d"),
        TimeSinceLast(name="uid_secs_since_last", entity=UID),
    ]
    distinct: list[Feature] = [
        DistinctCount(name=f"{ent_name}_nuniq_{val_name}_{w}", entity=ent, value=val, window=_W[w])
        for ent_name, ent, val_name, val in (
            ("uid", UID, "device", DEVICE),
            ("uid", UID, "email", EMAIL),
            ("uid", UID, "addr", ADDR),
            ("device", DEVICE, "uid", UID),
            ("email", EMAIL, "uid", UID),
        )
        for w in ("24h", "7d")
    ]
    first_seen: list[Feature] = [
        FirstSeen(name=f"uid_{n}_first_seen", entity=UID, value=v)
        for n, v in (("device", DEVICE), ("email", EMAIL), ("addr", ADDR))
    ]
    target: list[Feature] = [
        TargetEncoding(name=f"te_{c}", column=c) for c in (EMAIL, DEVICE, ADDR)
    ]
    return FeatureSet(
        features=tuple(request + velocity + amount + derived + distinct + first_seen + target),
        te_smoothing=te_smoothing,
    )
