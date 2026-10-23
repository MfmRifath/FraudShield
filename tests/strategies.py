"""Hypothesis strategies producing small canonical transaction frames with lots of collisions."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from hypothesis import strategies as st

from fraudshield.adapters.base import finalize_canonical
from fraudshield.config import LabelDelayConfig

T0 = datetime(2020, 1, 1, tzinfo=UTC)
# Short delays so labels become visible inside the generated time span.
TEST_DELAY = LabelDelayConfig(kind="uniform", min_days=0.05, max_days=2.0, seed=3)
EXTRA_NAMES = ("C1",)

# Offsets mix arbitrary seconds with multiples of every window length (10m, 1h, 24h; 7d = 7*24h)
# so events land *exactly* one window apart; duplicates exercise the tie-break path.
_offsets = st.one_of(
    st.integers(0, 3_600),
    st.integers(0, 9 * 86_400),
    st.integers(0, 12).map(lambda k: k * 600),
    st.integers(0, 30).map(lambda k: k * 3_600),
    st.integers(0, 8).map(lambda k: k * 86_400),
)


@st.composite
def raw_rows(draw: st.DrawFn, min_size: int = 1, max_size: int = 40) -> list[dict[str, Any]]:
    n = draw(st.integers(min_size, max_size))
    rows = []
    for i in range(n):
        rows.append(
            {
                "txn_id": f"t{i:03d}",
                "event_time": T0 + timedelta(seconds=draw(_offsets)),
                "uid": draw(st.sampled_from(["u0", "u1", "u2", "u3"])),
                "amount": draw(st.integers(0, 50_000)) / 100,
                "product": draw(st.sampled_from([None, "W", "C"])),
                "p_email_domain": draw(st.sampled_from([None, "gmail.com", "mail.ru", "x.org"])),
                "r_email_domain": draw(st.sampled_from([None, "gmail.com", "mail.ru"])),
                "device_type": draw(st.sampled_from([None, "mobile", "desktop"])),
                "device_info": draw(st.sampled_from([None, "d0", "d1", "d2"])),
                "addr": draw(st.sampled_from([None, "10", "20", "30"])),
                "label": draw(st.sampled_from([None, 0, 0, 1])),
                "x_c1": draw(st.one_of(st.none(), st.floats(-5, 5, allow_nan=False))),
            }
        )
    return rows


def to_canonical(rows: list[dict[str, Any]], delay: LabelDelayConfig = TEST_DELAY) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "txn_id": pl.String(),
        "event_time": pl.Datetime("us", "UTC"),
        "uid": pl.String(),
        "amount": pl.Float64(),
        "product": pl.String(),
        "p_email_domain": pl.String(),
        "r_email_domain": pl.String(),
        "device_type": pl.String(),
        "device_info": pl.String(),
        "addr": pl.String(),
        "label": pl.Int8(),
        "x_c1": pl.Float64(),
    }
    return finalize_canonical(pl.DataFrame(rows, schema=schema), EXTRA_NAMES, delay)


canonical_frames = raw_rows().map(to_canonical)


def assert_rows_close(
    got: list[dict[str, Any]], want: list[dict[str, Any]], names: list[str], tol: float = 1e-9
) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        for n in names:
            a, b = g[n], w[n]
            if a is None or b is None:
                assert (a, b) == (None, None), f"row {i} {n}: {a!r} vs {b!r}"
            elif isinstance(a, float) or isinstance(b, float):
                assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"row {i} {n}: {a} vs {b}"
            else:
                assert a == b, f"row {i} {n}: {a!r} vs {b!r}"
