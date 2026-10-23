"""Feature-set structure, and request features: scalar (online) == vectorised (offline)."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given

from fraudshield.adapters import get_adapter
from fraudshield.config import Settings
from fraudshield.features.definitions import (
    WindowAgg,
    build_feature_set,
)
from fraudshield.features.offline import compute_offline
from tests.oracle import oracle_features
from tests.strategies import EXTRA_NAMES, assert_rows_close, canonical_frames

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sparkov_sample.csv"


@pytest.fixture(scope="module")
def sparkov() -> pl.DataFrame:
    return get_adapter("sparkov", Settings()).load([FIXTURE])


@pytest.mark.unit
def test_feature_set_covers_every_spec_family() -> None:
    fs = build_feature_set(EXTRA_NAMES, 50.0)
    families = {f.family for f in fs.features}
    assert families == {
        "request",
        "velocity",
        "amount",
        "ratio",
        "recency",
        "distinct",
        "first_seen",
        "target_encoding",
    }
    names = set(fs.names)
    expected = {
        "amount", "log_amount", "amount_cents", "product", "card_network", "card_kind",
        "p_email_domain", "r_email_domain", "email_p_eq_r", "device_type", "hour", "weekday",
        "x_c1",
        "uid_cnt_10m", "uid_cnt_1h", "uid_cnt_24h", "uid_cnt_7d",
        *(f"uid_amt_{a}_{w}" for a in ("sum", "mean", "max") for w in ("1h", "24h", "7d")),
        "amt_to_uid_mean_7d", "uid_secs_since_last",
        *(f"uid_nuniq_{v}_{w}" for v in ("device", "email", "addr") for w in ("24h", "7d")),
        *(f"{e}_nuniq_uid_{w}" for e in ("device", "email") for w in ("24h", "7d")),
        "uid_device_first_seen", "uid_email_first_seen", "uid_addr_first_seen",
        "te_p_email_domain", "te_device_info", "te_addr",
    }  # fmt: skip
    assert names == expected
    assert set(fs.request_only) <= names
    assert all(n in fs.request_only for n in ("amount", "hour", "x_c1"))
    assert "uid_cnt_1h" not in fs.request_only


@pytest.mark.unit
def test_spec_hash_is_stable_and_sensitive() -> None:
    a = build_feature_set(EXTRA_NAMES, 50.0)
    assert a.spec_hash() == build_feature_set(EXTRA_NAMES, 50.0).spec_hash()
    assert a.spec_hash() != build_feature_set(EXTRA_NAMES, 10.0).spec_hash()
    assert a.spec_hash() != build_feature_set((), 50.0).spec_hash()


@pytest.mark.unit
def test_invalid_specs_are_rejected() -> None:
    from datetime import timedelta

    with pytest.raises(ValueError, match="value must be None"):
        WindowAgg(name="x", family="amount", entity="uid", agg="sum", window=timedelta(hours=1))
    fs = build_feature_set(EXTRA_NAMES, 50.0)
    with pytest.raises(ValueError, match="duplicate"):
        type(fs)(features=fs.features + fs.features[:1], te_smoothing=1.0)


def _request_scalar_vs_vector(df: pl.DataFrame, extra: tuple[str, ...]) -> None:
    fs = build_feature_set(extra, 50.0)
    request = fs.request_features
    vec = df.select([f.expr() for f in request]).to_dicts()
    for row, got in zip(df.to_dicts(), vec, strict=True):
        for f in request:
            a, b = f.scalar(row), got[f.name]
            if isinstance(a, float) and isinstance(b, float):
                assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (f.name, a, b)
            else:
                assert a == b, (f.name, a, b)


@pytest.mark.unit
@given(df=canonical_frames)
def test_request_scalar_matches_vectorised_generated(df: pl.DataFrame) -> None:
    _request_scalar_vs_vector(df, EXTRA_NAMES)


@pytest.mark.unit
def test_request_scalar_matches_vectorised_on_sparkov(sparkov: pl.DataFrame) -> None:
    _request_scalar_vs_vector(sparkov, Settings().datasets.sparkov.extra_columns)


@pytest.mark.unit
@pytest.mark.slow
def test_offline_matches_oracle_on_real_sparkov_prefix(sparkov: pl.DataFrame) -> None:
    """Real data, real feature set: the first 400 fixture events against the brute-force oracle."""
    fs = build_feature_set(Settings().datasets.sparkov.extra_columns, 50.0)
    head = sparkov.head(400)
    got = compute_offline(head, fs).select(fs.names).to_dicts()
    assert_rows_close(got, oracle_features(head.to_dicts(), fs), fs.names)
