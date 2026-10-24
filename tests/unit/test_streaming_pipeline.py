"""In-process pipeline parity: API (Redis reader, no inline update) -> decision event -> processor.

Each request is followed by a publisher flush whose sink hands the decision event (after its
JSON round trip) to the processor, i.e. the streaming topology with zero lag. Served features
must equal offline features (target encodings excluded: no labels flow here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakeredis
import orjson
import pytest
from fastapi.testclient import TestClient

from fraudshield.features.offline import compute_offline
from fraudshield.features.online import OnlineSpec
from fraudshield.features.redis_store import RedisFeatureStore, SyncRedisFeatureWriter
from fraudshield.pipeline import feature_set_for
from fraudshield.serving.app import create_app
from fraudshield.streaming.processor import Processor
from fraudshield.streaming.shadow import shadow_score
from fraudshield.synthetic import synthetic_canonical
from fraudshield.training.costs import CostConfig
from tests.serving_support import REPO, make_bundle, payload, serving_settings


class ProcessorSink:
    """Sink that applies decisions to Redis like the processor worker would."""

    def __init__(self, proc: Processor) -> None:
        self.proc = proc
        self.decisions: list[dict[str, Any]] = []

    async def send(self, topic: str, key: str, value: bytes) -> None:
        event = orjson.loads(value)
        if topic == "decisions":
            self.decisions.append(event)
            self.proc.on_decision(event)

    async def close(self) -> None:
        return None


@pytest.mark.unit
def test_api_processor_redis_pipeline_matches_offline(tmp_path: Path) -> None:
    s0 = serving_settings(tmp_path)
    bundle = make_bundle(s0, tmp_path / "bundle")
    # parity, not latency: a loaded CI box can exceed the 10 ms production feature budget on
    # fakeredis and (correctly) degrade a request, which is not what this test is about
    s = serving_settings(
        tmp_path, bundle_dir=bundle, update_store_inline=False, challenger_bundle_dir=None,
        feature_timeout_ms=2000.0,
    )  # fmt: skip
    fs = feature_set_for(s, "sparkov")
    spec = OnlineSpec.from_feature_set(fs)
    server = fakeredis.FakeServer()
    reader = RedisFeatureStore(spec, fakeredis.FakeAsyncRedis(server=server))
    sink = ProcessorSink(
        Processor(SyncRedisFeatureWriter(spec, fakeredis.FakeRedis(server=server)))
    )
    canon = synthetic_canonical(10_000, seed=0).head(800)
    with TestClient(create_app(s, store=reader, sink=sink)) as c:
        publisher = c.app.state.fs.service.publisher  # type: ignore[attr-defined]
        for row in canon.to_dicts():
            assert c.post("/score", json=payload(row)).json()["degraded"] is False
            c.portal.call(publisher.flush)  # type: ignore[union-attr]
    served = {e["txn_id"]: e["features"] for e in sink.decisions}
    assert len(served) == canon.height
    names = [n for n in fs.names if not n.startswith("te_")]
    worst = 0.0
    for off in compute_offline(canon, fs).to_dicts():
        on = served[off["txn_id"]]
        for n in names:
            a, b = on[n], off[n]
            if a is None or b is None:
                assert (a, b) == (None, None), (off["txn_id"], n, a, b)
            elif isinstance(b, str):
                assert a == b
            else:
                worst = max(worst, abs(float(a) - float(b)))
    assert worst < 1e-6

    # shadow scoring of the same decision events with a second bundle
    challenger = make_bundle(s0, tmp_path / "challenger", rounds=10)
    from fraudshield.model_bundle import ModelBundle

    cb = ModelBundle.load(challenger)
    costs = CostConfig.load(REPO / "configs/costs.yaml")
    shadows = [shadow_score(e, cb, "c1", costs) for e in sink.decisions[:50]]
    assert all(0.0 <= x["challenger"]["probability"] <= 1.0 for x in shadows)
    assert all(x["champion"]["version"] == "synthetic-20x15" for x in shadows)
    assert all(
        x["agree"] == (x["challenger"]["decision"] == x["champion"]["decision"]) for x in shadows
    )
