"""Label emitter: releases ground-truth labels when replayed event time reaches their availability.

Consumes ``transactions.truth``. Each label's availability time comes from
``fraudshield.labels`` (the same deterministic delay as offline, ADR-0005). The watermark is the
latest event time seen; labels whose ``label_available_time <= watermark`` are emitted to
``labels`` as ``label_source="chargeback"``.
"""

from __future__ import annotations

import heapq
import logging
from datetime import datetime
from typing import Any

from fraudshield.config import LabelDelayConfig, Settings
from fraudshield.labels import label_available_time
from fraudshield.streaming.common import WorkerMetrics, json_topic, make_app, parse_time

log = logging.getLogger(__name__)


class LabelScheduler:
    def __init__(self, delay: LabelDelayConfig) -> None:
        self.delay = delay
        self._heap: list[tuple[datetime, int, dict[str, Any]]] = []
        self._seq = 0
        self.watermark: datetime | None = None

    def on_truth(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        t = parse_time(msg["event_time"])
        if t is None:
            return []
        lat = label_available_time(str(msg["txn_id"]), t, self.delay)
        label = {
            "txn_id": msg["txn_id"],
            "uid": msg["uid"],
            "label": int(msg["label"]),
            "label_source": "chargeback",
            "label_available_time": lat.isoformat(),
            "attrs": msg.get("attrs", {}),
        }
        heapq.heappush(self._heap, (lat, self._seq, label))
        self._seq += 1
        if self.watermark is None or t > self.watermark:
            self.watermark = t
        out = []
        while self._heap and self._heap[0][0] <= self.watermark:
            out.append(heapq.heappop(self._heap)[2])
        return out

    @property
    def pending(self) -> int:
        return len(self._heap)


def run(settings: Settings) -> None:
    st = settings.streaming
    metrics = WorkerMetrics("label_emitter")
    metrics.serve(st.metrics_port)
    app = make_app(settings, "fraudshield-label-emitter")
    truth, labels = (
        json_topic(app, settings, st.topic_truth),
        json_topic(app, settings, st.topic_labels),
    )
    scheduler = LabelScheduler(settings.labels.delay)

    def handle(value: dict[str, Any], key: Any, ts: int, headers: Any) -> list[dict[str, Any]]:
        out = scheduler.on_truth(value)
        metrics.observe(st.topic_truth, ts)
        return out

    sdf = app.dataframe(truth).apply(handle, expand=True, metadata=True)
    sdf.to_topic(labels, key=lambda v: str(v["uid"]))
    app.run()
