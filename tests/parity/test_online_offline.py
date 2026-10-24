"""Online (store + per-event executor) vs offline (vectorised) feature parity.

Spec: replay a 2,000-event fixture through the online path; max |diff| < 1e-6 per feature.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from hypothesis import given

from fraudshield.adapters import get_adapter
from fraudshield.config import Settings
from fraudshield.features.definitions import FeatureSet, build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.features.online import OnlineSpec
from fraudshield.features.parity import max_abs_diff, parity_report
from fraudshield.features.store import replay
from tests.strategies import EXTRA_NAMES, canonical_frames

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sparkov_sample.csv"
TOL = 1e-6


def _parity(df: pl.DataFrame, fs: FeatureSet) -> dict[str, float]:
    return parity_report(df, fs)


@pytest.mark.parity
def test_parity_on_2000_event_sparkov_fixture() -> None:
    s = Settings()
    df = get_adapter("sparkov", s).load([FIXTURE])
    assert df.height == 2000
    fs = build_feature_set(s.datasets.sparkov.extra_columns, s.features.target_encoding_smoothing)
    worst = _parity(df, fs)
    bad = {k: v for k, v in worst.items() if not v < TOL}
    assert not bad, f"features over tolerance: {bad}"


@pytest.mark.parity
@given(df=canonical_frames)
def test_parity_on_generated_frames(df: pl.DataFrame) -> None:
    """Generated data has devices/emails/ties/window-edge offsets that Sparkov lacks."""
    worst = _parity(df, build_feature_set(EXTRA_NAMES, 5.0))
    bad = {k: v for k, v in worst.items() if not v < TOL}
    assert not bad, f"features over tolerance: {bad}"


@pytest.mark.parity
def test_online_spec_requirements() -> None:
    spec = OnlineSpec.from_feature_set(build_feature_set(EXTRA_NAMES, 5.0))
    day = 86_400 * 10**6
    # per-event history only where window aggregates need values; distinct counts use the index
    assert spec.history_fields == {"uid": ("amount",)}
    assert spec.retention_us == {"uid": 7 * day}
    assert spec.distinct == {
        ("device_info", "uid"): (day, 7 * day),
        ("p_email_domain", "uid"): (day, 7 * day),
        ("uid", "addr"): (day, 7 * day),
        ("uid", "device_info"): (day, 7 * day),
        ("uid", "p_email_domain"): (day, 7 * day),
    }
    assert spec.first_seen == (("uid", "addr"), ("uid", "device_info"), ("uid", "p_email_domain"))
    assert spec.last_time == ("uid",)
    assert spec.te_columns == ("addr", "device_info", "p_email_domain")


# ------------------------------------------------------------------ Redis-backed store
def _redis_stores(fs: FeatureSet) -> tuple[Any, Any]:
    import fakeredis

    from fraudshield.features.redis_store import RedisFeatureStore, SyncRedisFeatureWriter

    server = fakeredis.FakeServer()
    spec = OnlineSpec.from_feature_set(fs)
    reader = RedisFeatureStore(spec, fakeredis.FakeAsyncRedis(server=server))
    writer = SyncRedisFeatureWriter(spec, fakeredis.FakeRedis(server=server))
    return reader, writer


@pytest.mark.parity
def test_parity_redis_store_on_2000_event_fixture() -> None:
    """Same replay as above through RedisFeatureStore (async reads + writes, fakeredis)."""
    s = Settings()
    df = get_adapter("sparkov", s).load([FIXTURE])
    fs = build_feature_set(s.datasets.sparkov.extra_columns, s.features.target_encoding_smoothing)
    reader, _ = _redis_stores(fs)
    offline = compute_offline(df, fs).select(fs.names).to_dicts()
    online = asyncio.run(replay(reader, df.to_dicts(), fs))
    bad = {k: v for k, v in max_abs_diff(online, offline, fs.names).items() if not v < TOL}
    assert not bad, f"features over tolerance: {bad}"


@pytest.mark.parity
@given(df=canonical_frames)
def test_parity_redis_reader_with_processor_writer(df: pl.DataFrame) -> None:
    """API-style async reads + processor-style sync writes share one Redis: still exact."""
    from fraudshield.features.online import compute_online

    fs = build_feature_set(EXTRA_NAMES, 5.0)
    reader, writer = _redis_stores(fs)
    offline = compute_offline(df, fs).select(fs.names).to_dicts()

    async def run() -> list[dict[str, Any]]:
        out = []
        for row in df.to_dicts():
            out.append(compute_online(row, await reader.fetch(row), fs))
            writer.update(row)
            if row["label"] is not None:
                attrs = {c: row[c] for c in reader.spec.te_columns}
                writer.add_label(int(row["label"]), row["label_available_time"], attrs)
        return out

    bad = {
        k: v for k, v in max_abs_diff(asyncio.run(run()), offline, fs.names).items() if not v < TOL
    }
    assert not bad, f"features over tolerance: {bad}"
