"""Redis-backed feature store (same state model and semantics as ``InMemoryFeatureStore``).

Keys (all built by ``_RedisOps`` so the async reader and the sync writer agree):

* ``fs:h:<entity>:<value>``      ZSET of history events (score = t_us, member = JSON incl. txn id),
                                 trimmed to the entity's retention, PEXPIRE retention + 1 day
* ``fs:d:<entity>:<col>:<value>`` ZSET distinct-count index: member = a value of ``col``, score =
                                 when it was last seen (µs); a window count is one ZCOUNT, so a
                                 popular email domain costs O(log n), not its whole history
* ``fs:last:<entity>:<value>``   last event time (µs)
* ``fs:seen:<entity>:<col>:<v>`` SET of values seen for a first-seen pair
* ``fs:te:<col>:<value>``        HASH ``<visible_day_us>:n`` / ``:p`` label counts per snapshot day
* ``fs:te:__all__``              same, over all labels

``fetch`` is a single pipelined round trip. Writes assume per-entity time order, which Kafka
partitioning by uid provides (ADR-0010).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from typing import Any

import orjson
import redis
import redis.asyncio as aioredis

from fraudshield.features.online import (
    DistinctPair,
    EntityState,
    HistoryEvent,
    OnlineSpec,
    epoch_us,
    snapshot_day_us,
    visible_day_us,
)
from fraudshield.features.store import FeatureStoreError

Row = Mapping[str, Any]
_DAY_MS = 86_400_000
TE_ALL = "fs:te:__all__"


def _s(v: Any) -> str:
    return str(v)


class _RedisOps:
    """Queues commands on a (sync or async) pipeline and decodes the results."""

    def __init__(self, spec: OnlineSpec) -> None:
        self.spec = spec

    @staticmethod
    def hist_key(entity: str, value: Any) -> str:
        return f"fs:h:{entity}:{_s(value)}"

    @staticmethod
    def distinct_key(entity: str, col: str, value: Any) -> str:
        return f"fs:d:{entity}:{col}:{_s(value)}"

    @staticmethod
    def last_key(entity: str, value: Any) -> str:
        return f"fs:last:{entity}:{_s(value)}"

    @staticmethod
    def seen_key(entity: str, col: str, value: Any) -> str:
        return f"fs:seen:{entity}:{col}:{_s(value)}"

    @staticmethod
    def te_key(col: str, value: Any) -> str:
        return f"fs:te:{col}:{_s(value)}"

    def queue_fetch(self, pipe: Any, row: Row) -> Callable[[list[Any]], EntityState]:
        """Queue every read for ``row``; return a decoder for the pipeline's results."""
        t_us = epoch_us(row["event_time"])
        snap = snapshot_day_us(t_us)
        # (consumes a pipeline result?, setter(state, result))
        steps: list[tuple[bool, Callable[[EntityState, Any], None]]] = []

        for entity in self.spec.history_fields:
            key = row.get(entity)
            if key is not None:
                lo = t_us - self.spec.retention_us[entity]
                pipe.zrangebyscore(self.hist_key(entity, key), f"({lo}", f"({t_us}")
            steps.append((key is not None, partial(_set_history, entity=entity)))

        for pair, windows in self.spec.distinct.items():
            entity, col = pair
            key, v = row.get(entity), row.get(col)
            k = self.distinct_key(entity, col, key)
            for w in windows:
                if key is not None:
                    pipe.zcount(k, f"({t_us - w}", f"({t_us}")
                steps.append((key is not None, partial(_set_distinct, slot=(entity, col, w))))
            queued = key is not None and v is not None
            if queued:
                pipe.zscore(k, _s(v))
            steps.append((queued, partial(_set_distinct_last, pair=pair)))

        for entity in self.spec.last_time:
            key = row.get(entity)
            if key is not None:
                pipe.get(self.last_key(entity, key))
            steps.append((key is not None, partial(_set_last, entity=entity)))

        for pair in self.spec.first_seen:
            entity, col = pair
            key, v = row.get(entity), row.get(col)
            queued = key is not None and v is not None
            if queued:
                pipe.sismember(self.seen_key(entity, col, key), _s(v))
            steps.append((queued, partial(_set_seen, pair=pair)))

        if self.spec.te_columns:
            for col in self.spec.te_columns:
                v = row.get(col)
                if v is not None:
                    pipe.hgetall(self.te_key(col, v))
                steps.append((v is not None, partial(_set_te, col=col, snapshot_us=snap)))
            pipe.hgetall(TE_ALL)
            steps.append((True, partial(_set_te_global, snapshot_us=snap)))

        def decode(results: list[Any]) -> EntityState:
            st = EntityState()
            it = iter(results)
            for takes_result, setter in steps:
                setter(st, next(it) if takes_result else None)
            return st

        return decode

    def queue_update(self, pipe: Any, row: Row, txn_id: str) -> None:
        t_us = epoch_us(row["event_time"])
        for entity, fields in self.spec.history_fields.items():
            key = row.get(entity)
            if key is None:
                continue
            ret = self.spec.retention_us[entity]
            k = self.hist_key(entity, key)
            member = orjson.dumps({"i": txn_id, "t": t_us, **{f: row.get(f) for f in fields}})
            pipe.zadd(k, {member: t_us})
            pipe.zremrangebyscore(k, "-inf", t_us - ret)
            pipe.pexpire(k, ret // 1000 + _DAY_MS)
        for (entity, col), windows in self.spec.distinct.items():
            key, v = row.get(entity), row.get(col)
            if key is None or v is None:
                continue
            k = self.distinct_key(entity, col, key)
            ret = windows[-1]
            pipe.zadd(k, {_s(v): t_us}, gt=True)  # keeps the latest sighting
            pipe.zremrangebyscore(k, "-inf", t_us - ret)
            pipe.pexpire(k, ret // 1000 + _DAY_MS)
        for entity in self.spec.last_time:
            key = row.get(entity)
            if key is not None:
                pipe.set(self.last_key(entity, key), t_us)
        for entity, col in self.spec.first_seen:
            key, v = row.get(entity), row.get(col)
            if key is not None and v is not None:
                pipe.sadd(self.seen_key(entity, col, key), _s(v))

    def queue_label(
        self, pipe: Any, label: int, label_available_time: datetime, attrs: Mapping[str, Any]
    ) -> None:
        vis = visible_day_us(epoch_us(label_available_time))
        keys = [
            self.te_key(c, attrs.get(c)) for c in self.spec.te_columns if attrs.get(c) is not None
        ]
        for k in [*keys, TE_ALL]:
            pipe.hincrby(k, f"{vis}:n", 1)
            pipe.hincrby(k, f"{vis}:p", int(label))


def _set_history(st: EntityState, r: Any, *, entity: str) -> None:
    st.history[entity] = [] if r is None else _decode_history(r)


def _set_distinct(st: EntityState, r: Any, *, slot: tuple[str, str, int]) -> None:
    st.distinct[slot] = 0 if r is None else int(r)


def _set_distinct_last(st: EntityState, r: Any, *, pair: DistinctPair) -> None:
    st.distinct_last[pair] = None if r is None else int(float(r))


def _set_last(st: EntityState, r: Any, *, entity: str) -> None:
    st.last_t_us[entity] = None if r is None else int(r)


def _set_seen(st: EntityState, r: Any, *, pair: tuple[str, str]) -> None:
    st.seen[pair] = bool(r)


def _set_te(st: EntityState, r: Any, *, col: str, snapshot_us: int) -> None:
    st.te[col] = _sum_counts(r, snapshot_us)


def _set_te_global(st: EntityState, r: Any, *, snapshot_us: int) -> None:
    st.te_global = _sum_counts(r, snapshot_us)


def _decode_history(members: list[bytes]) -> list[HistoryEvent]:
    out = []
    for m in members:
        d = orjson.loads(m)
        t = d.pop("t")
        d.pop("i", None)
        out.append(HistoryEvent(int(t), d))
    return out


def _sum_counts(
    h: Mapping[bytes | str, bytes | str] | None, snapshot_us: int
) -> tuple[float, float]:
    n = pos = 0.0
    for field, val in (h or {}).items():
        f = field.decode() if isinstance(field, bytes) else field
        day, kind = f.rsplit(":", 1)
        if int(day) <= snapshot_us:
            if kind == "n":
                n += float(val)
            else:
                pos += float(val)
    return n, pos


class RedisFeatureStore:
    """Async store used by the API (``redis.asyncio``)."""

    def __init__(self, spec: OnlineSpec, client: aioredis.Redis) -> None:
        self.spec = spec
        self.client = client
        self._ops = _RedisOps(spec)

    @classmethod
    def from_url(cls, spec: OnlineSpec, url: str, timeout_s: float) -> RedisFeatureStore:
        client = aioredis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)
        return cls(spec, client)

    async def fetch(self, row: Row) -> EntityState:
        try:
            pipe = self.client.pipeline(transaction=False)
            decode = self._ops.queue_fetch(pipe, row)
            return decode(await pipe.execute())
        except (redis.RedisError, OSError) as e:
            raise FeatureStoreError(f"redis fetch failed: {e}") from e

    async def update(self, row: Row) -> None:
        try:
            pipe = self.client.pipeline(transaction=False)
            self._ops.queue_update(pipe, row, str(row["txn_id"]))
            await pipe.execute()
        except (redis.RedisError, OSError) as e:
            raise FeatureStoreError(f"redis update failed: {e}") from e

    async def add_label(
        self, txn_id: str, label: int, label_available_time: datetime, attrs: Mapping[str, Any]
    ) -> None:
        try:
            pipe = self.client.pipeline(transaction=False)
            self._ops.queue_label(pipe, label, label_available_time, attrs)
            await pipe.execute()
        except (redis.RedisError, OSError) as e:
            raise FeatureStoreError(f"redis label write failed: {e}") from e

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except (redis.RedisError, OSError):
            return False

    async def distinct_members(
        self, pair: DistinctPair, key: Any, since_us: int, limit: int
    ) -> list[str]:
        k = self._ops.distinct_key(pair[0], pair[1], key)
        try:
            got = await self.client.zrevrangebyscore(k, "+inf", f"({since_us}", start=0, num=limit)
        except (redis.RedisError, OSError) as e:
            raise FeatureStoreError(f"redis read failed: {e}") from e
        return [m.decode() if isinstance(m, bytes) else str(m) for m in got]

    async def close(self) -> None:
        await self.client.aclose()


class SyncRedisFeatureWriter:
    """Synchronous writer for the stream processor (Quix callbacks are synchronous)."""

    def __init__(self, spec: OnlineSpec, client: redis.Redis) -> None:
        self.spec = spec
        self.client = client
        self._ops = _RedisOps(spec)

    @classmethod
    def from_url(cls, spec: OnlineSpec, url: str, timeout_s: float = 2.0) -> SyncRedisFeatureWriter:
        return cls(
            spec, redis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)
        )

    def update(self, row: Row) -> None:
        pipe = self.client.pipeline(transaction=False)
        self._ops.queue_update(pipe, row, str(row["txn_id"]))
        pipe.execute()

    def add_label(
        self, label: int, label_available_time: datetime, attrs: Mapping[str, Any]
    ) -> None:
        pipe = self.client.pipeline(transaction=False)
        self._ops.queue_label(pipe, label, label_available_time, attrs)
        pipe.execute()
