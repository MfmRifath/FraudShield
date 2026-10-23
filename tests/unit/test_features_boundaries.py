"""Example-based tests pinning the exact window / snapshot boundary semantics."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl
import pytest

from fraudshield.config import LabelDelayConfig
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline
from tests.strategies import EXTRA_NAMES, T0, to_canonical

FS = build_feature_set(EXTRA_NAMES, te_smoothing=1.0)
_BLANK: dict[str, Any] = {
    "uid": "u0",
    "amount": 10.0,
    "product": None,
    "p_email_domain": None,
    "r_email_domain": None,
    "device_type": None,
    "device_info": None,
    "addr": None,
    "label": None,
    "x_c1": None,
}


def _frame(*events: dict[str, Any]) -> pl.DataFrame:
    return to_canonical([{**_BLANK, "txn_id": f"t{i}", **e} for i, e in enumerate(events)])


def _last(df: pl.DataFrame, name: str) -> Any:
    return compute_offline(df, FS)[name][-1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "window"),
    [
        ("uid_cnt_10m", timedelta(minutes=10)),
        ("uid_cnt_1h", timedelta(hours=1)),
        ("uid_cnt_24h", timedelta(hours=24)),
        ("uid_cnt_7d", timedelta(days=7)),
    ],
)
def test_count_window_is_half_open(name: str, window: timedelta) -> None:
    exactly = _frame({"event_time": T0}, {"event_time": T0 + window})
    inside = _frame({"event_time": T0}, {"event_time": T0 + window - timedelta(microseconds=1)})
    assert _last(exactly, name) == 1.0  # event exactly `window` ago is excluded
    assert _last(inside, name) == 2.0


@pytest.mark.unit
@pytest.mark.parametrize("agg", ["sum", "mean", "max"])
def test_amount_window_is_half_open(agg: str) -> None:
    hour = timedelta(hours=1)
    exactly = _frame({"event_time": T0, "amount": 100.0}, {"event_time": T0 + hour, "amount": 1.0})
    assert _last(exactly, f"uid_amt_{agg}_1h") == 1.0
    inside = _frame(
        {"event_time": T0, "amount": 100.0},
        {"event_time": T0 + hour - timedelta(microseconds=1), "amount": 1.0},
    )
    assert _last(inside, f"uid_amt_{agg}_1h") == {"sum": 101.0, "mean": 50.5, "max": 100.0}[agg]


@pytest.mark.unit
def test_distinct_window_is_half_open_and_ignores_nulls() -> None:
    day = timedelta(hours=24)
    df = _frame(
        {"event_time": T0, "device_info": "d0"},
        {"event_time": T0 + timedelta(hours=1), "device_info": None},
        {"event_time": T0 + day, "device_info": "d1"},
    )
    assert _last(df, "uid_nuniq_device_24h") == 1.0  # d0 is exactly 24h old -> excluded
    df2 = _frame(
        {"event_time": T0 + timedelta(seconds=1), "device_info": "d0"},
        {"event_time": T0 + day, "device_info": "d1"},
    )
    assert _last(df2, "uid_nuniq_device_24h") == 2.0


@pytest.mark.unit
def test_first_seen_and_since_last() -> None:
    df = _frame(
        {"event_time": T0, "device_info": "d0"},
        {"event_time": T0 + timedelta(seconds=90), "device_info": "d0"},
        {"event_time": T0 + timedelta(days=30), "device_info": None},
    )
    out = compute_offline(df, FS)
    assert out["uid_device_first_seen"].to_list() == [1.0, 0.0, None]
    assert out["uid_secs_since_last"].to_list() == [None, 90.0, 30 * 86_400 - 90.0]


@pytest.mark.unit
def test_ties_are_ordered_by_txn_id_and_made_unique() -> None:
    df = to_canonical([{**_BLANK, "txn_id": t, "event_time": T0} for t in ("b", "a", "c")])
    assert df["txn_id"].to_list() == ["a", "b", "c"]
    deltas = (df["event_time"] - T0).dt.total_microseconds().to_list()
    assert deltas == [0, 1, 2]
    # the earlier-ordered tie is strictly before, so it counts in the later one's window
    assert compute_offline(df, FS)["uid_cnt_10m"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_target_encoding_uses_label_only_from_next_snapshot() -> None:
    # Fixed half-day delay: label of the 12:00 event becomes available at exactly 00:00 next day.
    delay = LabelDelayConfig(kind="fixed", min_days=0.5, max_days=0.5)
    noon = T0 + timedelta(hours=12)
    rows = [
        {**_BLANK, "txn_id": "a", "event_time": noon, "addr": "10", "label": 1},
        {**_BLANK, "txn_id": "b", "event_time": T0 + timedelta(hours=23, minutes=59), "addr": "10"},
        {**_BLANK, "txn_id": "c", "event_time": T0 + timedelta(days=1, seconds=1), "addr": "10"},
        {**_BLANK, "txn_id": "d", "event_time": T0 + timedelta(days=1, seconds=2), "addr": "20"},
    ]
    df = to_canonical(rows, delay)
    assert df["label_available_time"][0] == T0 + timedelta(days=1)
    te = compute_offline(df, FS)["te_addr"].to_list()
    # m=1: before availability no labels exist at all -> null; afterwards p0 = 1/1 = 1.0.
    # c (addr 10): (1 + 1*1.0) / (1 + 1) = 1.0 ; d (addr 20, unseen): p0 = 1.0
    assert te == [None, None, 1.0, 1.0]
