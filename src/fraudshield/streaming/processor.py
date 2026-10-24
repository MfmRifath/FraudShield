"""Stream processor: keeps the Redis feature store up to date.

* ``decisions`` -> append the scored transaction to entity history (``update``);
* ``labels``    -> count a chargeback in target encodings from its visible snapshot (``add_label``).
  Analyst labels share the topic but are skipped: offline features count chargebacks only, and
  online must match them (ADR-0013).

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
from fraudshield.serving.activity import SyncRedisActivityWriter
from fraudshield.serving.activity import summary as activity_summary
from fraudshield.streaming.common import WorkerMetrics, decode_row, json_topic, make_app, parse_time

log = logging.getLogger(__name__)


class Processor:
    def __init__(
        self, writer: SyncRedisFeatureWriter, activity: SyncRedisActivityWriter | None = None
    ) -> None:
        self.writer = writer
        self.activity = activity

    def on_decision(self, event: dict[str, Any]) -> None:
        row = decode_row(event["row"])
        self.writer.update(row)
        if self.activity is not None:
            self.activity.append(str(row["uid"]), activity_summary(event, event["row"]))

    def on_label(self, msg: dict[str, Any]) -> None:
        if msg.get("label_source") == "analyst":
            return
        lat = parse_time(msg.get("label_available_time"))
        if lat is None:
            raise ValueError(f"label {msg.get('txn_id')} has no label_available_time")
        self.writer.add_label(int(msg["label"]), lat, msg.get("attrs") or {})


def run(settings: Settings) -> None:
    st = settings.streaming
    spec = OnlineSpec.from_feature_set(feature_set_for(settings, settings.serving.dataset))
    writer = SyncRedisFeatureWriter.from_url(spec, st.redis_url)
    proc = Processor(writer, SyncRedisActivityWriter(writer.client))
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
