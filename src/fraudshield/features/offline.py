"""Offline (batch) feature executor: canonical frame -> point-in-time-correct feature frame.

Implements the semantics documented in ``features.definitions`` with:

* Polars ``rolling_*_by(..., closed="right").over(entity)`` for ``(t - w, t]`` window aggregates;
* DuckDB ``count(DISTINCT v) OVER (PARTITION BY e ORDER BY ts RANGE BETWEEN w-1 PRECEDING AND
  CURRENT ROW)`` on integer microsecond timestamps for distinct counts (same half-open window);
* an as-of join against cumulative daily label tables for target encodings.

Input must be a validated canonical frame (strictly increasing, unique ``event_time``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Final

import duckdb
import polars as pl

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

KEY_COLUMNS: Final = ("txn_id", "event_time", "uid", "label", "label_available_time")
_T: Final = "event_time"
_RID: Final = "_rid"


def _us(window: timedelta) -> int:
    return (window.days * 86_400 + window.seconds) * 1_000_000 + window.microseconds


def _masked(entity: str, expr: pl.Expr) -> pl.Expr:
    return pl.when(pl.col(entity).is_not_null()).then(expr)


def _window_expr(f: WindowAgg) -> pl.Expr:
    size = f"{_us(f.window)}us"
    if f.agg == "count":
        e = pl.col(_T).is_not_null().cast(pl.Float64).rolling_sum_by(_T, size, closed="right")
    else:
        if f.value is None:  # pragma: no cover - rejected by WindowAgg.__post_init__
            raise ValueError(f"{f.name}: {f.agg} needs a value column")
        v = pl.col(f.value).cast(pl.Float64)
        roll = {"sum": v.rolling_sum_by, "mean": v.rolling_mean_by, "max": v.rolling_max_by}
        e = roll[f.agg](_T, size, closed="right")
    return _masked(f.entity, e.over(f.entity)).cast(pl.Float64).alias(f.name)


def _first_seen_expr(f: FirstSeen) -> pl.Expr:
    first = (pl.int_range(pl.len()).over([f.entity, f.value]) == 0).cast(pl.Float64)
    return (
        pl.when(pl.col(f.entity).is_not_null() & pl.col(f.value).is_not_null())
        .then(first)
        .alias(f.name)
    )


def _since_last_expr(f: TimeSinceLast) -> pl.Expr:
    delta = pl.col(_T) - pl.col(_T).shift(1).over(f.entity)
    return _masked(f.entity, delta.dt.total_microseconds() / 1e6).cast(pl.Float64).alias(f.name)


def _distinct_counts(df: pl.DataFrame, specs: Sequence[DistinctCount]) -> pl.DataFrame:
    if not specs:
        return pl.DataFrame()
    cols = sorted({c for s in specs for c in (s.entity, s.value)})
    ev = df.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias(_RID),
        pl.col(_T).dt.epoch("us").alias("_ts"),
        *cols,
    )
    selects = ",\n".join(
        f'count(DISTINCT "{s.value}") OVER (PARTITION BY "{s.entity}" ORDER BY _ts '
        f"RANGE BETWEEN {_us(s.window) - 1} PRECEDING AND CURRENT ROW) "
        f'AS "{s.name}"'
        for s in specs
    )
    with duckdb.connect() as con:
        con.execute("SET TimeZone = 'UTC'")
        con.register("ev", ev)
        out = con.sql(f"SELECT {_RID}, {selects} FROM ev ORDER BY {_RID}").pl()  # noqa: S608
    if out.height != df.height or not out[_RID].equals(ev[_RID]):
        raise RuntimeError("duckdb distinct-count result is misaligned")
    masked = [
        pl.when(df[s.entity].is_not_null()).then(out[s.name].cast(pl.Float64)).alias(s.name)
        for s in specs
    ]
    return pl.select(masked)


def _ceil_day(col: str) -> pl.Expr:
    floor = pl.col(col).dt.truncate("1d")
    return pl.when(floor == pl.col(col)).then(floor).otherwise(floor + pl.duration(days=1))


def _cumulative_label_table(labeled: pl.DataFrame, by: str | None) -> pl.DataFrame:
    keys = ["_visible"] if by is None else [by, "_visible"]
    agg = (
        labeled.group_by(keys)
        .agg(_n=pl.len().cast(pl.Float64), _pos=pl.col("label").cast(pl.Float64).sum())
        .sort(keys[::-1])  # by _visible first, so each group is time-ordered for cum_sum/asof
    )
    cum = [pl.col("_n").cum_sum(), pl.col("_pos").cum_sum()]
    return agg.with_columns(cum if by is None else [c.over(by) for c in cum])


def _target_encodings(df: pl.DataFrame, specs: Sequence[TargetEncoding], m: float) -> pl.DataFrame:
    if not specs:
        return pl.DataFrame()
    events = df.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias(_RID),
        pl.col(_T).dt.truncate("1d").alias("_snap"),
        *{s.column for s in specs},
    )  # already sorted by _snap because event_time is sorted
    labeled = df.filter(pl.col("label").is_not_null()).with_columns(
        _visible=_ceil_day("label_available_time")
    )
    glob = events.join_asof(
        _cumulative_label_table(labeled, None).rename({"_n": "_n_all", "_pos": "_pos_all"}),
        left_on="_snap",
        right_on="_visible",
        strategy="backward",
    ).sort(_RID)
    p0 = pl.when(pl.col("_n_all") > 0).then(pl.col("_pos_all") / pl.col("_n_all"))
    prior = glob.select(p0.alias("_p0"))["_p0"]

    out: list[pl.Series] = []
    for s in specs:
        table = _cumulative_label_table(labeled.filter(pl.col(s.column).is_not_null()), s.column)
        joined = (
            events.select(_RID, "_snap", s.column)
            .with_columns(_p0=prior)
            .join_asof(
                table,
                left_on="_snap",
                right_on="_visible",
                by=s.column,
                strategy="backward",
                check_sortedness=False,  # both sides sorted on the as-of key above
            )
            .sort(_RID)
        )
        n_c = pl.col("_n").fill_null(0.0)
        pos_c = pl.col("_pos").fill_null(0.0)
        te = (pos_c + m * pl.col("_p0")) / (n_c + m)
        out.append(
            joined.select(pl.when(pl.col(s.column).is_not_null()).then(te).alias(s.name))[s.name]
        )
    return pl.DataFrame(out)


def compute_offline(df: pl.DataFrame, fs: FeatureSet) -> pl.DataFrame:
    """Compute all features for every event of a canonical frame.

    Returns ``KEY_COLUMNS`` followed by ``fs.names`` in definition order, one row per input row,
    in input (time) order.
    """
    t = df[_T]
    if df.height and not (t.is_sorted() and t.is_unique().all()):
        raise ValueError("event_time must be strictly increasing (use a canonical frame)")

    exprs: dict[str, pl.Expr] = {}
    for f in fs.features:
        if isinstance(f, RequestFeature):
            exprs[f.name] = f.expr()
        elif isinstance(f, WindowAgg):
            exprs[f.name] = _window_expr(f)
        elif isinstance(f, FirstSeen):
            exprs[f.name] = _first_seen_expr(f)
        elif isinstance(f, TimeSinceLast):
            exprs[f.name] = _since_last_expr(f)
    base = df.select(*KEY_COLUMNS, *exprs.values())

    distinct = _distinct_counts(df, fs.of_type(DistinctCount))
    target = _target_encodings(df, fs.of_type(TargetEncoding), fs.te_smoothing)
    out = pl.concat([f for f in (base, distinct, target) if f.width], how="horizontal")

    ratios = [
        pl.when(pl.col(r.denominator) != 0)
        .then(pl.col(r.numerator) / pl.col(r.denominator))
        .cast(pl.Float64)
        .alias(r.name)
        for r in fs.of_type(Ratio)
    ]
    if ratios:
        out = out.with_columns(ratios)
    return out.select(*KEY_COLUMNS, *fs.names)
