"""Point-in-time correctness of the offline executor (hypothesis property tests)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fraudshield.features.definitions import FeatureSet, build_feature_set
from fraudshield.features.offline import KEY_COLUMNS, compute_offline
from tests.oracle import oracle_features
from tests.strategies import EXTRA_NAMES, T0, assert_rows_close, canonical_frames, to_canonical

FS: FeatureSet = build_feature_set(EXTRA_NAMES, te_smoothing=5.0)


def _feature_rows(df: pl.DataFrame) -> list[dict[str, object]]:
    return compute_offline(df, FS).select(FS.names).to_dicts()


@pytest.mark.unit
@given(df=canonical_frames)
def test_offline_matches_bruteforce_oracle(df: pl.DataFrame) -> None:
    assert_rows_close(_feature_rows(df), oracle_features(df.to_dicts(), FS), FS.names)


@pytest.mark.unit
@given(df=canonical_frames, data=st.data())
def test_no_future_leakage_prefix_invariance(df: pl.DataFrame, data: st.DataObject) -> None:
    """Features of the first k events must not change when later events are removed."""
    k = data.draw(st.integers(1, df.height))
    full = _feature_rows(df)[:k]
    prefix = _feature_rows(df.head(k))
    assert_rows_close(full, prefix, FS.names, tol=0.0)


@pytest.mark.unit
@given(df=canonical_frames, data=st.data())
def test_no_future_leakage_perturbed_future(df: pl.DataFrame, data: st.DataObject) -> None:
    """Rewriting every attribute of events after index k leaves features of 0..k unchanged."""
    k = data.draw(st.integers(0, df.height - 1))
    future = pl.int_range(pl.len()) > k
    perturbed = df.with_columns(
        amount=pl.when(future).then(pl.col("amount") * 3 + 1).otherwise(pl.col("amount")),
        uid=pl.when(future).then(pl.lit("u0")).otherwise(pl.col("uid")),
        device_info=pl.when(future).then(pl.lit("d9")).otherwise(pl.col("device_info")),
        p_email_domain=pl.when(future)
        .then(pl.lit("gmail.com"))
        .otherwise(pl.col("p_email_domain")),
        label=pl.when(future & pl.col("label").is_not_null())
        .then(1 - pl.col("label"))
        .otherwise(pl.col("label")),
    )
    assert_rows_close(_feature_rows(df)[: k + 1], _feature_rows(perturbed)[: k + 1], FS.names, 0.0)


@pytest.mark.unit
@given(df=canonical_frames, data=st.data())
def test_target_encoding_ignores_immature_labels(df: pl.DataFrame, data: st.DataObject) -> None:
    """Flipping labels that are not yet available at event i's snapshot must not change event i."""
    i = data.draw(st.integers(0, df.height - 1))
    t: datetime = df["event_time"][i]
    snap = t.replace(hour=0, minute=0, second=0, microsecond=0)
    immature = pl.col("label").is_not_null() & (pl.col("label_available_time") > snap)
    flipped = df.with_columns(
        label=pl.when(immature).then(1 - pl.col("label")).otherwise(pl.col("label"))
    )
    te = [f.name for f in FS.by_family("target_encoding")]
    a = compute_offline(df, FS).select(te).row(i)
    b = compute_offline(flipped, FS).select(te).row(i)
    assert a == b


@pytest.mark.unit
@given(df=canonical_frames)
def test_offline_is_deterministic_and_shaped(df: pl.DataFrame) -> None:
    a, b = compute_offline(df, FS), compute_offline(df, FS)
    assert a.equals(b)
    assert a.columns == [*KEY_COLUMNS, *FS.names]
    assert a.height == df.height
    assert a["txn_id"].to_list() == df["txn_id"].to_list()
    # the current event is always inside its own windows
    assert (a["uid_cnt_10m"] >= 1).all()


@pytest.mark.unit
def test_rejects_unsorted_input() -> None:
    base = {"uid": "u0", "amount": 1.0, "label": None, "x_c1": None}
    rows = [
        {**base, "txn_id": "a", "event_time": T0},
        {**base, "txn_id": "b", "event_time": T0 + timedelta(seconds=5)},
    ]
    df = to_canonical([{**_NULL_ROW, **r} for r in rows])
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_offline(df.reverse(), FS)


_NULL_ROW = dict.fromkeys(
    ["product", "p_email_domain", "r_email_domain", "device_type", "device_info", "addr"]
)
