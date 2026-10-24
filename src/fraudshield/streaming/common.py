"""Shared plumbing for the stream workers: Quix app factory, metrics, event decoding."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
from quixstreams import Application
from quixstreams.models import TopicConfig
from quixstreams.models.topics import Topic

from fraudshield.config import Settings

log = logging.getLogger(__name__)


def make_app(settings: Settings, consumer_group: str) -> Application:
    st = settings.streaming
    return Application(
        broker_address=st.bootstrap_servers,
        consumer_group=consumer_group,
        auto_offset_reset="earliest",
        auto_create_topics=True,
        commit_interval=1.0,
        state_dir=Path("/tmp/quix-state") / consumer_group,  # noqa: S108 - workers are stateless
        loglevel="WARNING",
    )


def json_topic(app: Application, settings: Settings, name: str) -> Topic:
    cfg = TopicConfig(num_partitions=settings.streaming.partitions, replication_factor=1)
    return app.topic(
        name, value_deserializer="json", value_serializer="json", key_serializer="str", config=cfg
    )


class WorkerMetrics:
    """Per-worker Prometheus metrics; ``consumer_lag_seconds`` = now - Kafka message timestamp."""

    def __init__(self, worker: str) -> None:
        self.worker = worker
        self.registry = CollectorRegistry()
        self.lag = Gauge(
            "consumer_lag_seconds",
            "Age of the last consumed message",
            ["worker", "topic"],
            registry=self.registry,
        )
        self.processed = Counter(
            "messages_processed_total",
            "Messages handled",
            ["worker", "topic", "outcome"],
            registry=self.registry,
        )

    def observe(
        self, topic: str, timestamp_ms: int, outcome: str = "ok", now_s: float | None = None
    ) -> float:
        lag = max(0.0, (time.time() if now_s is None else now_s) - timestamp_ms / 1000.0)
        self.lag.labels(worker=self.worker, topic=topic).set(lag)
        self.processed.labels(worker=self.worker, topic=topic, outcome=outcome).inc()
        return lag

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)
        log.info("%s metrics on :%d", self.worker, port)


def parse_time(v: Any) -> datetime | None:
    if v is None or isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


def decode_row(row: dict[str, Any]) -> dict[str, Any]:
    """A canonical row that went through JSON: restore datetimes."""
    out = dict(row)
    out["event_time"] = parse_time(row["event_time"])
    out["label_available_time"] = parse_time(row.get("label_available_time"))
    return out
