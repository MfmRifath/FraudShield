"""Case store contract (in-memory), the activity log and the processor's activity write."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import fakeredis
import orjson
import pytest

from fraudshield.cases.store import InMemoryCaseStore
from fraudshield.config import Settings
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.online import OnlineSpec
from fraudshield.features.redis_store import RedisFeatureStore, SyncRedisFeatureWriter
from fraudshield.serving.activity import (
    FIELDS,
    InMemoryActivityLog,
    RedisActivityLog,
    SyncRedisActivityWriter,
    summary,
)
from fraudshield.streaming.processor import Processor
from fraudshield.streaming.replayer import load_source
from tests.case_contract import check_case_store

T0 = datetime(2021, 3, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def fixture_rows() -> list[dict[str, object]]:
    return load_source(Settings(), "fixture", "sparkov").to_dicts()


@pytest.mark.unit
def test_in_memory_case_store_contract() -> None:
    asyncio.run(check_case_store(InMemoryCaseStore()))


def _entry(i: int) -> dict[str, object]:
    return {"txn_id": f"t{i}", "event_time": (T0 + timedelta(minutes=i)).isoformat(), "amount": i}


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["memory", "redis"])
def test_activity_log_newest_first_and_bounded(kind: str) -> None:
    log = (
        InMemoryActivityLog(keep=3)
        if kind == "memory"
        else RedisActivityLog(fakeredis.FakeAsyncRedis(), keep=3)
    )

    async def run() -> tuple[
        list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
    ]:
        for i in range(5):
            await log.append("u1", _entry(i))
        await log.append("u2", _entry(9))
        return await log.recent("u1", 10), await log.recent("u1", 2), await log.recent("nobody", 5)

    all_, two, none = asyncio.run(run())
    assert [e["txn_id"] for e in all_] == ["t4", "t3", "t2"]
    assert [e["txn_id"] for e in two] == ["t4", "t3"]
    assert none == []


@pytest.mark.unit
def test_activity_summary_prefers_event_fields_and_keeps_only_listed_fields() -> None:
    row = {
        "txn_id": "t1",
        "amount": 5.0,
        "device_info": "d1",
        "secret_feature": 1.0,
        "decision": "x",
    }
    event = {"decision": "review", "probability": 0.7, "model_version": "m1", "degraded": False}
    out = summary(event, row)
    assert tuple(out) == FIELDS
    assert (out["decision"], out["device_info"], out["addr"]) == ("review", "d1", None)


@pytest.mark.unit
def test_processor_appends_activity_that_the_api_reads(
    fixture_rows: list[dict[str, object]],
) -> None:
    s = Settings()
    spec = OnlineSpec.from_feature_set(build_feature_set(s.datasets.sparkov.extra_columns, 50.0))
    server = fakeredis.FakeServer()
    sync = fakeredis.FakeRedis(server=server)
    proc = Processor(SyncRedisFeatureWriter(spec, sync), SyncRedisActivityWriter(sync))
    row = fixture_rows[0]
    event = orjson.loads(orjson.dumps(
        {"row": row, "txn_id": row["txn_id"], "decision": "approve", "probability": 0.01}, default=str
    ))  # fmt: skip
    proc.on_decision(event)
    reader = RedisActivityLog(fakeredis.FakeAsyncRedis(server=server))
    got = asyncio.run(reader.recent(str(row["uid"]), 5))
    assert [(g["txn_id"], g["decision"]) for g in got] == [(row["txn_id"], "approve")]
    assert got[0]["amount"] == row["amount"]


@pytest.mark.unit
def test_processor_counts_chargebacks_but_not_analyst_labels(
    fixture_rows: list[dict[str, object]],
) -> None:
    """Online TE must match offline (chargebacks only); an analyst label must not crash it."""
    s = Settings()
    spec = OnlineSpec.from_feature_set(build_feature_set(s.datasets.sparkov.extra_columns, 50.0))
    server = fakeredis.FakeServer()
    proc = Processor(SyncRedisFeatureWriter(spec, fakeredis.FakeRedis(server=server)))
    reader = RedisFeatureStore(spec, fakeredis.FakeAsyncRedis(server=server))
    row = fixture_rows[0]
    t = row["event_time"]
    assert isinstance(t, datetime)
    probe = dict(row, event_time=t + timedelta(days=2), txn_id="probe")
    before = asyncio.run(reader.fetch(probe)).te_global
    # what the API publishes for an analyst label (no label_available_time, no attrs)
    proc.on_label({"txn_id": row["txn_id"], "label": 1, "label_source": "analyst",
                   "label_time": (t + timedelta(hours=1)).isoformat(), "analyst_id": "ana"})  # fmt: skip
    assert asyncio.run(reader.fetch(probe)).te_global == before
    proc.on_label({"txn_id": row["txn_id"], "label": 1, "label_source": "chargeback",
                   "label_available_time": (t + timedelta(hours=1)).isoformat(),
                   "attrs": {"addr": row["addr"]}})  # fmt: skip
    assert asyncio.run(reader.fetch(probe)).te_global != before
