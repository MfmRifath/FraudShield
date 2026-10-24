"""Stream-worker logic without Kafka (Kafka/Redis end-to-end lives in tests/integration)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import orjson
import polars as pl
import pytest
import requests

from fraudshield.config import LabelDelayConfig, Settings
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.online import OnlineSpec
from fraudshield.features.redis_store import RedisFeatureStore, SyncRedisFeatureWriter
from fraudshield.labels import label_available_time
from fraudshield.streaming.common import WorkerMetrics, decode_row
from fraudshield.streaming.gateway import Gateway
from fraudshield.streaming.label_emitter import LabelScheduler
from fraudshield.streaming.parquet_sink import HourlyParquetWriter
from fraudshield.streaming.processor import Processor
from fraudshield.streaming.replayer import (
    load_source,
    namespaced,
    pacing_delay,
    raw_message,
    replay,
    select_window,
    truth_message,
)
from tests.strategies import EXTRA_NAMES

T0 = datetime(2019, 3, 1, tzinfo=UTC)


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, dict[str, Any]]] = []
        self.flushed = False

    def produce(self, topic: str, value: bytes, key: bytes) -> None:
        self.sent.append((topic, key, orjson.loads(value)))

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        self.flushed = True
        return 0


@pytest.fixture(scope="module")
def fixture_rows() -> list[dict[str, Any]]:
    return load_source(Settings(), "fixture", "sparkov").to_dicts()


# ------------------------------------------------------------------ replayer
@pytest.mark.unit
def test_raw_message_has_no_labels_and_maps_extras(fixture_rows: list[dict[str, Any]]) -> None:
    msg = raw_message(fixture_rows[0])
    assert "label" not in msg
    assert "label_available_time" not in msg
    assert set(msg["extra"]) == {"city_pop", "lat", "long", "merch_lat", "merch_long"}
    from fraudshield.serving.schemas import ScoreRequest

    ScoreRequest.model_validate(msg)  # the gateway can post it as-is


@pytest.mark.unit
def test_replay_produces_in_order_keyed_by_uid(fixture_rows: list[dict[str, Any]]) -> None:
    df = select_window(pl.DataFrame(fixture_rows), None, None, 50)
    sleeps: list[float] = []
    prod = FakeProducer()
    stats = replay(df.to_dicts(), prod, Settings(), rate=100, namespace="run1", sleep=sleeps.append)
    raw = [(k, v) for t, k, v in prod.sent if t == "transactions.raw"]
    truth = [v for t, _, v in prod.sent if t == "transactions.truth"]
    assert stats.raw == 50 == len(raw)
    assert stats.truth == len(truth) == 50  # every fixture row is labelled
    assert all(k == v["uid"].encode() for k, v in raw)
    assert all(v["uid"].startswith("run1:") and v["txn_id"].startswith("run1:") for _, v in raw)
    times = [v["event_time"] for _, v in raw]
    assert times == sorted(times)
    assert sleeps == [pytest.approx(0.01)] * 50
    assert prod.flushed


@pytest.mark.unit
def test_select_window_and_namespace() -> None:
    df = pl.DataFrame({"event_time": [T0 + timedelta(hours=h) for h in (3, 1, 2)], "x": [3, 1, 2]})
    out = select_window(df, T0 + timedelta(hours=1), T0 + timedelta(hours=3), None)
    assert out["x"].to_list() == [1, 2]
    row = {
        "txn_id": "t",
        "uid": "u",
        "device_info": None,
        "p_email_domain": "g",
        "addr": "1",
        "amount": 5.0,
    }
    ns = namespaced(row, "n")
    assert (ns["txn_id"], ns["uid"], ns["device_info"], ns["p_email_domain"], ns["amount"]) == (
        "n:t",
        "n:u",
        None,
        "n:g",
        5.0,
    )
    assert namespaced(row, None) == row


@pytest.mark.unit
def test_pacing_delay() -> None:
    assert pacing_delay(None, T0, speed=60, rate=0) == 0
    assert pacing_delay(T0, T0 + timedelta(minutes=2), speed=60, rate=0) == pytest.approx(2.0)
    assert pacing_delay(T0, T0 + timedelta(seconds=1), speed=60, rate=10) == pytest.approx(0.1)
    assert truth_message({"label": None}, ()) is None


# ------------------------------------------------------------------ label emitter
@pytest.mark.unit
def test_label_scheduler_releases_at_availability_time() -> None:
    delay = LabelDelayConfig(kind="fixed", min_days=1.0, max_days=1.0)
    sch = LabelScheduler(delay)

    def truth(i: int, t: datetime) -> dict[str, Any]:
        return {
            "txn_id": f"t{i}",
            "uid": "u",
            "event_time": t.isoformat(),
            "label": 1,
            "attrs": {"addr": "1"},
        }

    assert sch.on_truth(truth(0, T0)) == []
    assert sch.on_truth(truth(1, T0 + timedelta(hours=23))) == []
    out = sch.on_truth(truth(2, T0 + timedelta(days=1)))  # watermark reaches t0 + 1 day exactly
    assert [o["txn_id"] for o in out] == ["t0"]
    assert out[0]["label_source"] == "chargeback"
    assert out[0]["label_available_time"] == label_available_time("t0", T0, delay).isoformat()
    assert sch.pending == 2


# ------------------------------------------------------------------ processor + redis
@pytest.mark.unit
def test_processor_writes_what_the_reader_sees(fixture_rows: list[dict[str, Any]]) -> None:
    import asyncio

    s = Settings()
    fs = build_feature_set(s.datasets.sparkov.extra_columns, 50.0)
    spec = OnlineSpec.from_feature_set(fs)
    server = fakeredis.FakeServer()
    proc = Processor(SyncRedisFeatureWriter(spec, fakeredis.FakeRedis(server=server)))
    reader = RedisFeatureStore(spec, fakeredis.FakeAsyncRedis(server=server))
    row = fixture_rows[0]
    # decision events carry the canonical row through JSON (datetimes become ISO strings)
    event = orjson.loads(orjson.dumps({"row": row}, default=str))
    assert decode_row(event["row"])["event_time"] == row["event_time"]
    proc.on_decision(event)
    later = dict(row, event_time=row["event_time"] + timedelta(minutes=1), txn_id="next")
    state = asyncio.run(reader.fetch(later))
    assert len(state.history["uid"]) == 1
    assert state.last_t_us["uid"] is not None
    proc.on_label({"label": 1, "label_available_time": (row["event_time"] + timedelta(hours=1)).isoformat(),
                   "attrs": {"addr": row["addr"]}})  # fmt: skip
    tomorrow = dict(later, event_time=row["event_time"] + timedelta(days=1))
    st2 = asyncio.run(reader.fetch(tomorrow))
    assert st2.te_global == (1.0, 1.0)
    assert st2.te["addr"] == (1.0, 1.0)


# ------------------------------------------------------------------ gateway
class _Resp:
    def __init__(self, code: int) -> None:
        self.status_code = code
        self.text = ""


class _Session:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls = 0

    def post(self, url: str, json: Any, timeout: float) -> _Resp:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "outcome", "calls"),
    [
        ([200], "ok", 1),
        ([422], "client_error", 1),
        ([503, 200], "ok", 2),
        ([requests.ConnectionError("down")] * 4, "failed", 4),
    ],
)
def test_gateway_retries(script: list[Any], outcome: str, calls: int) -> None:
    session = _Session(list(script))
    gw = Gateway("http://api:8000/", 1.0, session=session, retries=3, sleep=lambda s: None)  # type: ignore[arg-type]
    assert gw.url == "http://api:8000/score"
    assert gw.handle({"txn_id": "t"}) == outcome
    assert session.calls == calls


# ------------------------------------------------------------------ parquet sink
@pytest.mark.unit
def test_hourly_parquet_writer(tmp_path: Path) -> None:
    now = [0.0]
    w = HourlyParquetWriter(tmp_path, flush_s=60, max_rows=3, clock=lambda: now[0])

    def ev(i: int, t: datetime) -> dict[str, Any]:
        return {"txn_id": f"t{i}", "uid": "u", "event_time": t.isoformat(), "amount": 1.0, "decision": "approve",
                "probability": 0.1, "model_version": "v", "variant": "champion", "degraded": False,
                "latency_ms": 1.0, "rules_fired": [], "reason_codes": [], "features": {"amount": 1.0}}  # fmt: skip

    assert w.add(ev(0, T0)) == []
    assert w.add(ev(1, T0 + timedelta(hours=1))) == []
    files = w.add(ev(2, T0 + timedelta(hours=1, minutes=5)))  # max_rows reached
    assert sorted(f.parent.name for f in files) == ["hour=00", "hour=01"]
    assert w.rows_written == 3
    w.add(ev(3, T0))
    now[0] = 61.0
    assert len(w.add(ev(4, T0))) == 1  # time-based flush
    df = pl.read_parquet(tmp_path / "decisions" / "**" / "*.parquet")
    assert df.height == 5
    assert orjson.loads(df["features"][0]) == {"amount": 1.0}
    assert w.flush() == []


@pytest.mark.unit
def test_parquet_timer_flushes_an_idle_stream(tmp_path: Path) -> None:
    """Regression: buffered rows must reach disk without waiting for the next message."""
    import time as _time

    w = HourlyParquetWriter(tmp_path, flush_s=0.05, max_rows=10_000)
    w.add({"txn_id": "t0", "uid": "u", "event_time": T0.isoformat(), "decision": "approve"})
    w.start_timer()
    deadline = _time.monotonic() + 5
    while w.rows_written == 0 and _time.monotonic() < deadline:
        _time.sleep(0.02)
    assert w.rows_written == 1
    assert w.close() == []


# ------------------------------------------------------------------ metrics
@pytest.mark.unit
def test_worker_lag_metric() -> None:
    m = WorkerMetrics("w")
    assert m.observe("topic", timestamp_ms=1_000_000, now_s=1_030.5) == pytest.approx(30.5)
    from prometheus_client import generate_latest

    text = generate_latest(m.registry).decode()
    assert 'consumer_lag_seconds{topic="topic",worker="w"} 30.5' in text
    assert 'messages_processed_total{outcome="ok",topic="topic",worker="w"} 1.0' in text


@pytest.mark.unit
def test_streaming_cli_rejects_unknown_worker() -> None:
    from fraudshield.streaming.__main__ import main

    assert main(["nope"]) == 2


@pytest.mark.unit
def test_feature_set_for_tests_is_redis_compatible() -> None:
    spec = OnlineSpec.from_feature_set(build_feature_set(EXTRA_NAMES, 5.0))
    assert spec.history_fields["uid"]
