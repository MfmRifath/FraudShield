"""End to end through docker compose (``make test-integration``).

replayer -> transactions.raw -> gateway -> API /score (Redis reads) -> decisions ->
processor (Redis writes) / parquet sink / shadow (@challenger) -> shadow.scores.

The run is namespaced (ids and entity values prefixed) so it is isolated from earlier runs in
the same Redis. Events are paced (20/s) so the processor's update of event N lands before
event N+1 of the same card is scored; served features must then equal offline features.
"""

from __future__ import annotations

import time
import urllib.request
import uuid
from typing import Any

import orjson
import polars as pl
import pytest
import redis
from confluent_kafka import Consumer, Producer

from fraudshield.config import Settings
from fraudshield.features.offline import compute_offline
from fraudshield.pipeline import feature_set_for
from fraudshield.streaming.replayer import NAMESPACED, load_source, replay, select_window

N_EVENTS = 200
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def run(settings: Settings) -> dict[str, Any]:
    ns = f"it{uuid.uuid4().hex[:8]}"
    canon = select_window(load_source(settings, "fixture", "sparkov"), None, None, N_EVENTS)
    producer = Producer({"bootstrap.servers": settings.streaming.bootstrap_servers})
    stats = replay(canon.to_dicts(), producer, settings, rate=20, namespace=ns)
    return {"ns": ns, "canon": canon, "stats": stats}


def _consume(
    settings: Settings, topic: str, ns: str, want: int, timeout_s: float = 120
) -> list[dict[str, Any]]:
    c = Consumer({"bootstrap.servers": settings.streaming.bootstrap_servers, "group.id": f"{ns}-{topic}-{uuid.uuid4().hex[:6]}",
                  "auto.offset.reset": "earliest", "enable.auto.commit": False})  # fmt: skip
    c.subscribe([topic])
    out: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_s
    try:
        while len(out) < want and time.monotonic() < deadline:
            msg = c.poll(1.0)
            if msg is None or msg.error():
                continue
            v = orjson.loads(msg.value())
            if str(v.get("txn_id", "")).startswith(f"{ns}:"):
                out[v["txn_id"]] = v
    finally:
        c.close()
    return list(out.values())


def _metrics(host: str) -> str:
    with urllib.request.urlopen(f"http://{host}:9100/metrics", timeout=5) as r:
        return str(r.read().decode())


def test_every_event_is_scored_with_full_features(settings: Settings, run: dict[str, Any]) -> None:
    assert run["stats"].raw == N_EVENTS
    decisions = _consume(settings, settings.streaming.topic_decisions, run["ns"], N_EVENTS)
    assert len(decisions) == N_EVENTS
    assert all(d["degraded"] is False for d in decisions), [
        d["txn_id"] for d in decisions if d["degraded"]
    ][:5]
    assert {d["model_version"] for d in decisions} == {"dev-fixture-50"}
    run["decisions"] = decisions


def test_served_features_match_offline(settings: Settings, run: dict[str, Any]) -> None:
    decisions = run.get("decisions") or _consume(
        settings, settings.streaming.topic_decisions, run["ns"], N_EVENTS
    )
    ns = run["ns"]
    canon: pl.DataFrame = run["canon"].with_columns(
        [
            pl.when(pl.col(c).is_not_null()).then(pl.lit(f"{ns}:") + pl.col(c)).alias(c)
            for c in NAMESPACED
        ]
    )
    fs = feature_set_for(settings, "sparkov")
    offline = {r["txn_id"]: r for r in compute_offline(canon, fs).to_dicts()}
    names = [
        n for n in fs.names if not n.startswith("te_")
    ]  # TE also counts labels of earlier runs
    mismatches = []
    for d in decisions:
        off = offline[d["txn_id"]]
        for n in names:
            a, b = d["features"][n], off[n]
            same = (a is None and b is None) or (
                a is not None
                and b is not None
                and (a == b if isinstance(b, str) else abs(float(a) - float(b)) < 1e-6)
            )
            if not same:
                mismatches.append((d["txn_id"], n, a, b))
    assert not mismatches, mismatches[:10]


def test_processor_updated_redis(settings: Settings, run: dict[str, Any]) -> None:
    r = redis.from_url(settings.streaming.redis_url)
    uids = {f"{run['ns']}:{u}" for u in run["canon"]["uid"].unique().to_list()}
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        sizes = [r.zcard(f"fs:h:uid:{u}") for u in uids]
        if all(s > 0 for s in sizes):
            break
        time.sleep(1)
    assert all(r.exists(f"fs:last:uid:{u}") for u in uids)


def test_shadow_scores_every_decision(settings: Settings, run: dict[str, Any]) -> None:
    shadows = _consume(settings, settings.streaming.topic_shadow, run["ns"], N_EVENTS)
    assert len(shadows) == N_EVENTS
    assert {s["challenger"]["version"] for s in shadows} == {"dev-fixture-80"}
    assert all(0.0 <= s["challenger"]["probability"] <= 1.0 for s in shadows)


def test_worker_metrics_expose_consumer_lag(settings: Settings, run: dict[str, Any]) -> None:
    for host, topic in (("processor", "decisions"), ("gateway", "transactions.raw"), ("shadow", "decisions"),
                        ("parquet-sink", "decisions"), ("label-emitter", "transactions.truth")):  # fmt: skip
        text = _metrics(host)
        lines = [
            ln
            for ln in text.splitlines()
            if ln.startswith("consumer_lag_seconds{") and f'topic="{topic}"' in ln
        ]
        assert lines, f"{host}: no consumer_lag_seconds for {topic}"
        assert float(lines[0].rsplit(" ", 1)[1]) < 30.0, lines[0]
