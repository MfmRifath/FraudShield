"""Kafka sink for the API publisher (confluent-kafka / librdkafka).

``produce`` only enqueues into librdkafka's buffer; delivery happens on its own thread, so the
publisher's drain task never blocks on the network. A full local buffer raises BufferError,
which the publisher logs and counts as a dropped event.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from confluent_kafka import KafkaError, Message, Producer

log = logging.getLogger(__name__)


class KafkaSink:
    def __init__(self, bootstrap_servers: str, **config: Any) -> None:
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "linger.ms": 5,
                "compression.type": "lz4",
                "queue.buffering.max.messages": 100_000,
                **config,
            }
        )
        self.delivery_errors = 0

    def _on_delivery(self, err: KafkaError | None, msg: Message) -> None:
        if err is not None:
            self.delivery_errors += 1
            log.warning("kafka delivery to %s failed: %s", msg.topic(), err)

    async def send(self, topic: str, key: str, value: bytes) -> None:
        self._producer.produce(topic, value=value, key=key.encode(), on_delivery=self._on_delivery)
        self._producer.poll(0)

    async def close(self) -> None:
        remaining = await asyncio.to_thread(self._producer.flush, 5.0)
        if remaining:
            log.warning("%d kafka messages not delivered at shutdown", remaining)
