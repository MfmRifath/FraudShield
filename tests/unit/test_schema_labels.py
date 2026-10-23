from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandera.errors
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from fraudshield.config import LabelDelayConfig
from fraudshield.labels import label_delay_us, unit_hash, with_label_available_time
from fraudshield.schema import Transaction, from_records, to_records, validate_canonical
from tests.strategies import canonical_frames

_US_PER_DAY = 86_400 * 1_000_000


# ------------------------------------------------------------------ labels
@pytest.mark.unit
@given(txn_id=st.text(min_size=1, max_size=20), seed=st.integers(0, 2**31))
def test_unit_hash_in_unit_interval_and_stable(txn_id: str, seed: int) -> None:
    u = unit_hash(txn_id, seed)
    assert 0.0 <= u < 1.0
    assert u == unit_hash(txn_id, seed)


@pytest.mark.unit
@given(
    txn_id=st.text(min_size=1, max_size=20),
    lo=st.floats(0, 10, allow_nan=False),
    width=st.floats(0, 10, allow_nan=False),
)
def test_uniform_delay_within_bounds(txn_id: str, lo: float, width: float) -> None:
    cfg = LabelDelayConfig(kind="uniform", min_days=lo, max_days=lo + width)
    d = label_delay_us(txn_id, cfg)
    assert d >= 1
    assert round(lo * _US_PER_DAY) - 1 <= d <= round((lo + width) * _US_PER_DAY) + 1


@pytest.mark.unit
def test_fixed_delay_and_known_hash_value() -> None:
    assert (
        label_delay_us("x", LabelDelayConfig(kind="fixed", min_days=2, max_days=5))
        == 2 * _US_PER_DAY
    )
    assert label_delay_us("x", LabelDelayConfig(kind="fixed", min_days=0, max_days=0)) == 1
    # pin the hash so a silent change (e.g. digest size) breaks label timing reproducibility
    assert unit_hash("2987000", 17) == 0.007473537408903719


@pytest.mark.unit
def test_label_time_only_for_labelled_rows() -> None:
    t = datetime(2020, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {"txn_id": ["a", "b"], "event_time": [t, t], "label": [1, None]},
        schema={"txn_id": pl.String, "event_time": pl.Datetime("us", "UTC"), "label": pl.Int8},
    )
    out = with_label_available_time(df, LabelDelayConfig(kind="fixed", min_days=1, max_days=1))
    assert out["label_available_time"].to_list() == [t + timedelta(days=1), None]


# ------------------------------------------------------------------ canonical schema
@pytest.mark.unit
@given(df=canonical_frames)
def test_records_roundtrip(df: pl.DataFrame) -> None:
    back = from_records(to_records(df))
    assert back.equals(df)


@pytest.mark.unit
def test_transaction_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Transaction(txn_id="a", event_time=datetime(2020, 1, 1), uid="u", amount=1.0)  # noqa: DTZ001


@pytest.mark.unit
def test_transaction_converts_to_utc() -> None:
    from zoneinfo import ZoneInfo

    t = datetime(2020, 1, 1, 5, 30, tzinfo=ZoneInfo("Asia/Colombo"))
    assert Transaction(txn_id="a", event_time=t, uid="u", amount=1.0).event_time.tzinfo == UTC


def _valid() -> pl.DataFrame:
    t = datetime(2020, 1, 1, tzinfo=UTC)
    return from_records(
        [
            Transaction(txn_id="a", event_time=t, uid="u", amount=1.0),
            Transaction(
                txn_id="b",
                event_time=t + timedelta(seconds=1),
                uid="u",
                amount=2.0,
                label=1,
                label_available_time=t + timedelta(days=1),
            ),
        ]
    )


@pytest.mark.unit
def test_validate_accepts_valid_frame() -> None:
    validate_canonical(_valid())


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.reverse(), id="unsorted"),
        pytest.param(lambda d: d.with_columns(txn_id=pl.lit("a")), id="dup-txn"),
        pytest.param(
            lambda d: d.with_columns(event_time=pl.col("event_time").min()), id="dup-time"
        ),
        pytest.param(lambda d: d.with_columns(amount=-pl.col("amount")), id="neg-amount"),
        pytest.param(lambda d: d.with_columns(label=pl.lit(2, pl.Int8)), id="bad-label"),
        pytest.param(
            lambda d: d.with_columns(label_available_time=pl.col("event_time")), id="label-time"
        ),
        pytest.param(lambda d: d.with_columns(x_bad=pl.lit("s")), id="non-float-extra"),
        pytest.param(lambda d: d.with_columns(unexpected=pl.lit(1.0)), id="extra-column"),
    ],
)
def test_validate_rejects(mutate: object) -> None:
    bad = mutate(_valid())  # type: ignore[operator]
    with pytest.raises((pandera.errors.SchemaError, ValueError)):
        validate_canonical(bad)
