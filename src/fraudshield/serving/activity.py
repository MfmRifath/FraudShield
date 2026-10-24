"""Recent activity per card (``uid``) for the case timeline in the analyst UI.

In-process (``InMemoryActivityLog``) when the API updates state itself; Redis
(``RedisActivityLog``: API reads, processor writes from the decisions topic) in the stack.
Newest first; each card keeps the last ``keep`` events (TTL 30 days in Redis).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Final, Protocol

import orjson
import redis
import redis.asyncio as aioredis

FIELDS: Final = ("txn_id", "event_time", "amount", "decision", "probability", "product",
                 "device_info", "p_email_domain", "addr", "degraded", "model_version")  # fmt: skip
_TTL_S: Final = 30 * 86_400


def summary(event: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Activity entry from a decision event (+ the canonical row it was scored from)."""
    merged = {**row, **event}
    return {k: merged.get(k) for k in FIELDS}


def _key(uid: str) -> str:
    return f"fs:activity:{uid}"


class ActivityLog(Protocol):
    async def append(self, uid: str, entry: dict[str, Any]) -> None: ...

    async def recent(self, uid: str, n: int) -> list[dict[str, Any]]: ...


class InMemoryActivityLog:
    def __init__(self, keep: int = 50) -> None:
        self._log: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=keep))

    async def append(self, uid: str, entry: dict[str, Any]) -> None:
        self._log[uid].appendleft(entry)

    async def recent(self, uid: str, n: int) -> list[dict[str, Any]]:
        return list(self._log.get(uid, ()))[:n]


class RedisActivityLog:
    def __init__(self, client: aioredis.Redis, keep: int = 50) -> None:
        self.client = client
        self.keep = keep

    async def append(self, uid: str, entry: dict[str, Any]) -> None:
        pipe = self.client.pipeline(transaction=False)
        pipe.lpush(_key(uid), orjson.dumps(entry, default=str))
        pipe.ltrim(_key(uid), 0, self.keep - 1)
        pipe.expire(_key(uid), _TTL_S)
        await pipe.execute()

    async def recent(self, uid: str, n: int) -> list[dict[str, Any]]:
        return [orjson.loads(x) for x in await self.client.lrange(_key(uid), 0, n - 1)]


class SyncRedisActivityWriter:
    """Processor-side writer (Quix callbacks are synchronous)."""

    def __init__(self, client: redis.Redis, keep: int = 50) -> None:
        self.client = client
        self.keep = keep

    def append(self, uid: str, entry: dict[str, Any]) -> None:
        pipe = self.client.pipeline(transaction=False)
        pipe.lpush(_key(uid), orjson.dumps(entry, default=str))
        pipe.ltrim(_key(uid), 0, self.keep - 1)
        pipe.expire(_key(uid), _TTL_S)
        pipe.execute()
