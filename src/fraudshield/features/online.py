"""Online (per-event) feature executor.

Interprets the same specs as ``features.offline`` with identical semantics (see the
``features.definitions`` docstring). The work is split in two:

* ``OnlineSpec``: derived from a ``FeatureSet``, it says what per-entity state a store must keep
  (history fields + retention per entity, seen-sets, last-event times, target-encoding columns).
* ``compute_online(row, state, ...)``: a pure function from the current canonical row plus the
  ``EntityState`` read from the store (events strictly *before* the current one) to features.

Stores (``serving.store``) only persist and return state; all feature logic lives here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from fraudshield.features.definitions import (
    DistinctCount,
    FeatureSet,
    FirstSeen,
    Ratio,
    RequestFeature,
    TargetEncoding,
    TimeSinceLast,
    WindowAgg,
)

US_PER_DAY: Final = 86_400 * 1_000_000
Row = Mapping[str, Any]


def epoch_us(t: datetime) -> int:
    """Exact integer microseconds since the epoch (no float rounding)."""
    delta = t - datetime(1970, 1, 1, tzinfo=t.tzinfo)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def snapshot_day_us(t_us: int) -> int:
    """00:00 UTC of the event's day: the target-encoding snapshot an event may use."""
    return (t_us // US_PER_DAY) * US_PER_DAY


def visible_day_us(label_available_us: int) -> int:
    """First snapshot (midnight) at or after a label's availability time."""
    return -((-label_available_us) // US_PER_DAY) * US_PER_DAY


def _window_us(w: Any) -> int:
    return int((w.days * 86_400 + w.seconds) * 1_000_000 + w.microseconds)


@dataclass(frozen=True)
class OnlineSpec:
    """Per-entity state requirements of a feature set."""

    history_fields: dict[str, tuple[str, ...]]  # entity column -> value columns stored per event
    retention_us: dict[str, int]  # entity column -> longest window that reads its history
    first_seen: tuple[tuple[str, str], ...]  # (entity, value) pairs needing a seen-set
    last_time: tuple[str, ...]  # entities needing the last event time
    te_columns: tuple[str, ...]

    @classmethod
    def from_feature_set(cls, fs: FeatureSet) -> OnlineSpec:
        fields_: dict[str, set[str]] = {}
        retention: dict[str, int] = {}
        for f in fs.features:
            if isinstance(f, WindowAgg | DistinctCount):
                needed = fields_.setdefault(f.entity, set())
                value = f.value
                if value is not None:
                    needed.add(value)
                retention[f.entity] = max(retention.get(f.entity, 0), _window_us(f.window))
        return cls(
            history_fields={e: tuple(sorted(v)) for e, v in sorted(fields_.items())},
            retention_us=retention,
            first_seen=tuple(sorted({(f.entity, f.value) for f in fs.of_type(FirstSeen)})),
            last_time=tuple(sorted({f.entity for f in fs.of_type(TimeSinceLast)})),
            te_columns=tuple(sorted({f.column for f in fs.of_type(TargetEncoding)})),
        )


@dataclass(frozen=True, slots=True)
class HistoryEvent:
    t_us: int
    values: Mapping[str, Any]


@dataclass
class EntityState:
    """What a store returns for one event (all of it from events strictly before that event)."""

    history: dict[str, list[HistoryEvent]] = field(default_factory=dict)
    last_t_us: dict[str, int | None] = field(default_factory=dict)
    seen: dict[tuple[str, str], bool] = field(default_factory=dict)
    te: dict[str, tuple[float, float]] = field(default_factory=dict)  # column -> (n, pos)
    te_global: tuple[float, float] = (0.0, 0.0)


def _agg(agg: str, values: Sequence[float]) -> float:
    if agg == "count":
        return float(len(values))
    if agg == "sum":
        return math.fsum(values)
    if agg == "mean":
        return math.fsum(values) / len(values)
    return max(values)


def compute_online(row: Row, state: EntityState, fs: FeatureSet) -> dict[str, Any]:
    """Features for the current event ``row`` (a canonical row mapping incl. ``x_*`` extras)."""
    t_us = epoch_us(row["event_time"])
    out: dict[str, Any] = {}
    for f in fs.features:
        if isinstance(f, RequestFeature):
            out[f.name] = f.scalar(row)
        elif isinstance(f, WindowAgg | DistinctCount):
            if row.get(f.entity) is None:
                out[f.name] = None
                continue
            lo = t_us - _window_us(f.window)
            prior = [e for e in state.history.get(f.entity, ()) if lo < e.t_us < t_us]
            if isinstance(f, WindowAgg):
                if f.value is None:
                    out[f.name] = float(len(prior) + 1)
                else:
                    vals = [float(e.values[f.value]) for e in prior] + [float(row[f.value])]
                    out[f.name] = _agg(f.agg, vals)
            else:
                seen = {e.values.get(f.value) for e in prior}
                seen.add(row.get(f.value))
                seen.discard(None)
                out[f.name] = float(len(seen))
        elif isinstance(f, FirstSeen):
            if row.get(f.entity) is None or row.get(f.value) is None:
                out[f.name] = None
            else:
                out[f.name] = 0.0 if state.seen.get((f.entity, f.value), False) else 1.0
        elif isinstance(f, TimeSinceLast):
            last = state.last_t_us.get(f.entity)
            if row.get(f.entity) is None or last is None:
                out[f.name] = None
            else:
                out[f.name] = (t_us - last) / 1e6
        elif isinstance(f, TargetEncoding):
            out[f.name] = _te(
                row.get(f.column),
                state.te.get(f.column, (0.0, 0.0)),
                state.te_global,
                fs.te_smoothing,
            )
    for r in fs.of_type(Ratio):
        num, den = out[r.numerator], out[r.denominator]
        out[r.name] = None if num is None or den is None or den == 0 else num / den
    return out


def _te(
    value: Any, counts: tuple[float, float], glob: tuple[float, float], m: float
) -> float | None:
    if value is None or glob[0] <= 0:
        return None
    p0 = glob[1] / glob[0]
    n_c, pos_c = counts
    return (pos_c + m * p0) / (n_c + m)
