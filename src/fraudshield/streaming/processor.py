"""Stream processor: keeps the Redis feature store up to date.

* ``decisions`` -> append the scored transaction to entity history (``update``);
* ``labels``    -> count the label in target encodings from its visible snapshot (``add_label``).

With the processor running, the API sets ``serving.update_store_inline=false``. Per-entity
order holds because every topic is keyed by uid (ADR-0010).
"""

from __future__ import annotations

import logging
from typing import Any

from fraudshield.config import Settings
from fraudshield.features.online import OnlineSpec
from fraudshield.features.redis_store import SyncRedisFeatureWriter
from fraudshield.pipeline import feature_set_for
from fraudshield.streaming.common import WorkerMetrics, decode_row, json_topic, make_app, parse_time

log = logging.getLogger(__name__)


class Processor:
    def __init__(self, writer: SyncRedisFeatureWriter) -> None:
        self.writer = writer

    def on_decision(self, event: dict[str, Any]) -> None:
        self.writer.update(decode_row(event["row"]))

    def on_label(self, msg: dict[str, Any]) -> None:
        lat = parse_time(msg["label_available_time"])
        if lat is None:
            raise ValueError(f"label {msg.get('txn_id')} has no label_available_time")
        self.writer.add_label(int(msg["label"]), lat, msg.get("attrs") or {})


def run(settings: Settings) -> None:
    st = settings.streaming
    spec = OnlineSpec.from_feature_set(feature_set_for(settings, settings.serving.dataset))
    proc = Processor(SyncRedisFeatureWriter.from_url(spec, st.redis_url))
    metrics = WorkerMetrics("processor")
    metrics.serve(st.metrics_port)
    app = make_app(settings, "fraudshield-processor")

    def on_decision(value: dict[str, Any], key: Any, ts: int, headers: Any) -> None:
        proc.on_decision(value)
        metrics.observe(st.topic_decisions, ts)

    def on_label(value: dict[str, Any], key: Any, ts: int, headers: Any) -> None:
        proc.on_label(value)
        metrics.observe(st.topic_labels, ts)

    app.dataframe(json_topic(app, settings, st.topic_decisions)).update(on_decision, metadata=True)
    app.dataframe(json_topic(app, settings, st.topic_labels)).update(on_label, metadata=True)
    app.run()
