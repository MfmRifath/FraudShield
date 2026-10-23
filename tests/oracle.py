"""Brute-force reference implementation of the feature semantics in features/definitions.py.

Deliberately naive (O(n^2), pure Python, no Polars/DuckDB) and written straight from the
docstring so it can serve as an independent oracle for the vectorised executors.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

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

Row = dict[str, Any]


def _midnight(t: datetime) -> datetime:
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


def oracle_features(rows: list[Row], fs: FeatureSet) -> list[dict[str, Any]]:
    """``rows`` are canonical rows sorted by unique event_time."""
    out: list[dict[str, Any]] = []
    for i, cur in enumerate(rows):
        t: datetime = cur["event_time"]
        hist = rows[: i + 1]  # strictly earlier events + the current one
        feats: dict[str, Any] = {}
        for f in fs.features:
            if isinstance(f, RequestFeature):
                feats[f.name] = f.scalar(cur)
            elif isinstance(f, WindowAgg):
                feats[f.name] = _window(hist, cur, t, f)
            elif isinstance(f, DistinctCount):
                feats[f.name] = _distinct(hist, cur, t, f)
            elif isinstance(f, FirstSeen):
                feats[f.name] = _first_seen(rows[:i], cur, f)
            elif isinstance(f, TimeSinceLast):
                feats[f.name] = _since_last(rows[:i], cur, t, f)
            elif isinstance(f, TargetEncoding):
                feats[f.name] = _te(rows, cur, t, f, fs.te_smoothing)
        for r in fs.of_type(Ratio):
            num, den = feats[r.numerator], feats[r.denominator]
            feats[r.name] = None if den is None or den == 0 or num is None else num / den
        out.append(feats)
    return out


def _in_window(r: Row, t: datetime, w: timedelta) -> bool:
    return bool(t - w < r["event_time"] <= t)


def _window(hist: list[Row], cur: Row, t: datetime, f: WindowAgg) -> float | None:
    e = cur[f.entity]
    if e is None:
        return None
    sel = [r for r in hist if r[f.entity] == e and _in_window(r, t, f.window)]
    if f.agg == "count":
        return float(len(sel))
    vals = [float(r[f.value]) for r in sel if f.value is not None]
    if f.agg == "sum":
        return sum(vals)
    if f.agg == "mean":
        return sum(vals) / len(vals)
    return max(vals)


def _distinct(hist: list[Row], cur: Row, t: datetime, f: DistinctCount) -> float | None:
    e = cur[f.entity]
    if e is None:
        return None
    vals = {
        r[f.value]
        for r in hist
        if r[f.entity] == e and _in_window(r, t, f.window) and r[f.value] is not None
    }
    return float(len(vals))


def _first_seen(before: list[Row], cur: Row, f: FirstSeen) -> float | None:
    e, v = cur[f.entity], cur[f.value]
    if e is None or v is None:
        return None
    return 0.0 if any(r[f.entity] == e and r[f.value] == v for r in before) else 1.0


def _since_last(before: list[Row], cur: Row, t: datetime, f: TimeSinceLast) -> float | None:
    e = cur[f.entity]
    if e is None:
        return None
    prev = [r["event_time"] for r in before if r[f.entity] == e]
    return (t - prev[-1]).total_seconds() if prev else None


def _te(rows: list[Row], cur: Row, t: datetime, f: TargetEncoding, m: float) -> float | None:
    c = cur[f.column]
    if c is None:
        return None
    snap = _midnight(t)
    # Note: ALL rows are scanned (not just earlier ones): visibility is decided by label time only.
    visible = [
        r
        for r in rows
        if r["label"] is not None
        and r["label_available_time"] is not None
        and r["label_available_time"] <= snap
    ]
    if not visible:
        return None
    p0 = sum(int(r["label"]) for r in visible) / len(visible)
    mine = [r for r in visible if r[f.column] == c]
    return (sum(int(r["label"]) for r in mine) + m * p0) / (len(mine) + m)
