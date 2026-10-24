"""Non-blocking event publishing (Kafka topics) and SSE fan-out.

``/score`` must never wait on Kafka: events go into a bounded in-process queue that a background
task drains into a ``Sink``. When the queue is full the event is dropped and counted.
The Kafka sink arrives in Phase 4; ``MemorySink`` serves tests and the demo container.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any, Protocol

import orjson

log = logging.getLogger(__name__)


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, default=str, option=orjson.OPT_SERIALIZE_NUMPY)


class Sink(Protocol):
    async def send(self, topic: str, key: str, value: bytes) -> None: ...

    async def close(self) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, bytes]] = []

    async def send(self, topic: str, key: str, value: bytes) -> None:
        self.messages.append((topic, key, value))

    async def close(self) -> None:
        return None

    def decoded(self, topic: str) -> list[dict[str, Any]]:
        return [orjson.loads(v) for t, _, v in self.messages if t == topic]


class Publisher:
    """Bounded queue + background drain task."""

    def __init__(
        self, sink: Sink, maxsize: int, on_drop: Callable[[str], None] | None = None
    ) -> None:
        self.sink = sink
        self._queue: asyncio.Queue[tuple[str, str, bytes]] = asyncio.Queue(maxsize)
        self._task: asyncio.Task[None] | None = None
        self._on_drop = on_drop

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._drain(), name="publisher")

    def publish(self, topic: str, key: str, value: Any) -> bool:
        """Enqueue without awaiting. Returns False (and counts a drop) if the queue is full."""
        try:
            self._queue.put_nowait((topic, key, dumps(value)))
            return True
        except asyncio.QueueFull:
            if self._on_drop:
                self._on_drop(topic)
            return False

    async def _drain(self) -> None:
        while True:
            topic, key, value = await self._queue.get()
            try:
                await self.sink.send(topic, key, value)
            except Exception:
                log.exception("publish to %s failed; event dropped", topic)
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.flush(), timeout=5)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.sink.close()


class Broadcaster:
    """Fan-out of decision summaries to SSE subscribers (bounded; slow clients lose old events)."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self.maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        for q in self._subscribers:
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(event)

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)
