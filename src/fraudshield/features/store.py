"""Feature-store interface and the in-memory implementation.

A store persists exactly the state described by an ``OnlineSpec`` and hands it back as an
``EntityState`` for one event. Feature computation is not its job (see ``features.online``).

* ``fetch(row)``: state from events strictly before ``row`` (the caller then computes features).
* ``update(row)``: record ``row`` as history (called after scoring: by the API in single-process
  mode, or by the stream processor from the decisions topic).
* ``add_label(...)``: a label became known; it counts in target encodings from the first
  snapshot (00:00 UTC) at or after ``label_available_time``.
* ``distinct_members(...)``: values recently seen for an entity (the UI's linked cards).

``RedisFeatureStore`` (Phase 4) implements the same protocol with pipelined Redis reads.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

from fraudshield.features.online import (
    DistinctPair,
    EntityState,
    HistoryEvent,
    OnlineSpec,
    epoch_us,
    snapshot_day_us,
    visible_day_us,
)

Row = Mapping[str, Any]


class FeatureStoreError(RuntimeError):
    """Any failure to read or write feature state (callers fall back to the degraded path)."""


class FeatureStore(Protocol):
    spec: OnlineSpec

    async def fetch(self, row: Row) -> EntityState: ...

    async def update(self, row: Row) -> None: ...

    async def add_label(
        self, txn_id: str, label: int, label_available_time: datetime, attrs: Mapping[str, Any]
    ) -> None: ...

    async def ping(self) -> bool: ...

    async def distinct_members(
        self, pair: DistinctPair, key: Any, since_us: int, limit: int
    ) -> list[str]: ...


class TargetEncodingCounts:
    """Cumulative (n, pos) per column value and overall, as of a daily snapshot.

    Labels wait in a heap keyed by their visible day and are folded in as snapshots advance.
    Events arrive in time order in production and replay, so the common path is O(new labels);
    a request for an earlier snapshot is answered by a full recount.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns
        self._pending: list[tuple[int, int, int, tuple[Any, ...]]] = []
        self._all: list[tuple[int, int, tuple[Any, ...]]] = []
        self._seq = 0
        self._applied_day: int | None = None
        self._counts: dict[str, dict[Any, list[float]]] = {c: {} for c in columns}
        self._global = [0.0, 0.0]

    def add(self, label: int, label_available_us: int, attrs: Mapping[str, Any]) -> None:
        vis = visible_day_us(label_available_us)
        values = tuple(attrs.get(c) for c in self.columns)
        self._all.append((vis, label, values))
        if self._applied_day is not None and vis <= self._applied_day:
            self._apply(label, values)  # late label for an already-applied snapshot
        else:
            heapq.heappush(self._pending, (vis, self._seq, label, values))
            self._seq += 1

    def _apply(self, label: int, values: tuple[Any, ...]) -> None:
        self._global[0] += 1
        self._global[1] += label
        for c, v in zip(self.columns, values, strict=True):
            if v is not None:
                cell = self._counts[c].setdefault(v, [0.0, 0.0])
                cell[0] += 1
                cell[1] += label

    def lookup(
        self, snapshot_us: int, values: Mapping[str, Any]
    ) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
        if self._applied_day is not None and snapshot_us < self._applied_day:
            return self._recount(snapshot_us, values)
        while self._pending and self._pending[0][0] <= snapshot_us:
            _, _, label, vals = heapq.heappop(self._pending)
            self._apply(label, vals)
        self._applied_day = snapshot_us
        per = {}
        for c in self.columns:
            cell = self._counts[c].get(values.get(c))
            per[c] = (cell[0], cell[1]) if cell else (0.0, 0.0)
        return per, (self._global[0], self._global[1])

    def _recount(
        self, snapshot_us: int, values: Mapping[str, Any]
    ) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
        n = pos = 0.0
        per = {c: [0.0, 0.0] for c in self.columns}
        for vis, label, vals in self._all:
            if vis > snapshot_us:
                continue
            n += 1
            pos += label
            for c, v in zip(self.columns, vals, strict=True):
                if v is not None and v == values.get(c):
                    per[c][0] += 1
                    per[c][1] += label
        return {c: (a, b) for c, (a, b) in per.items()}, (n, pos)


class InMemoryFeatureStore:
    """Single-process store for tests, parity replay and the demo container."""

    def __init__(self, spec: OnlineSpec) -> None:
        self.spec = spec
        self._history: dict[str, dict[Any, deque[HistoryEvent]]] = {
            e: defaultdict(deque) for e in spec.history_fields
        }
        # (entity, value) -> entity key -> value -> last time seen (µs)
        self._distinct: dict[DistinctPair, dict[Any, dict[Any, int]]] = {
            p: defaultdict(dict) for p in spec.distinct
        }
        self._seen: dict[tuple[str, str], dict[Any, set[Any]]] = {
            p: defaultdict(set) for p in spec.first_seen
        }
        self._last: dict[str, dict[Any, int]] = {e: {} for e in spec.last_time}
        self._te = TargetEncodingCounts(spec.te_columns)
        self.healthy = True  # tests flip this to simulate an outage

    def _check(self) -> None:
        if not self.healthy:
            raise FeatureStoreError("in-memory store marked unhealthy")

    async def fetch(self, row: Row) -> EntityState:
        self._check()
        t_us = epoch_us(row["event_time"])
        state = EntityState()
        for entity in self.spec.history_fields:
            key = row.get(entity)
            state.history[entity] = [] if key is None else list(self._history[entity].get(key, ()))
        for pair, windows in self.spec.distinct.items():
            entity, value = pair
            key, v = row.get(entity), row.get(value)
            seen = self._distinct[pair].get(key, {}) if key is not None else {}
            for w in windows:
                lo = t_us - w
                state.distinct[(entity, value, w)] = sum(
                    1 for ts in seen.values() if lo < ts < t_us
                )
            state.distinct_last[pair] = None if v is None else seen.get(v)
        for entity in self.spec.last_time:
            key = row.get(entity)
            state.last_t_us[entity] = None if key is None else self._last[entity].get(key)
        for pair in self.spec.first_seen:
            entity, value = pair
            key, v = row.get(entity), row.get(value)
            state.seen[pair] = (
                key is not None and v is not None and v in self._seen[pair].get(key, ())
            )
        if self.spec.te_columns:
            state.te, state.te_global = self._te.lookup(snapshot_day_us(t_us), row)
        return state

    async def update(self, row: Row) -> None:
        self._check()
        t_us = epoch_us(row["event_time"])
        for entity, fields in self.spec.history_fields.items():
            key = row.get(entity)
            if key is None:
                continue
            q = self._history[entity][key]
            q.append(HistoryEvent(t_us, {f: row.get(f) for f in fields}))
            horizon = t_us - self.spec.retention_us[entity]
            while q and q[0].t_us <= horizon:
                q.popleft()
        for pair, windows in self.spec.distinct.items():
            entity, value = pair
            key, v = row.get(entity), row.get(value)
            if key is None or v is None:
                continue
            seen = self._distinct[pair][key]
            seen[v] = t_us
            horizon = t_us - windows[-1]
            for old in [x for x, ts in seen.items() if ts <= horizon]:
                del seen[old]
        for entity in self.spec.last_time:
            if (key := row.get(entity)) is not None:
                self._last[entity][key] = t_us
        for pair in self.spec.first_seen:
            entity, value = pair
            key, v = row.get(entity), row.get(value)
            if key is not None and v is not None:
                self._seen[pair][key].add(v)

    async def add_label(
        self, txn_id: str, label: int, label_available_time: datetime, attrs: Mapping[str, Any]
    ) -> None:
        self._check()
        self._te.add(int(label), epoch_us(label_available_time), attrs)

    async def ping(self) -> bool:
        return self.healthy

    async def distinct_members(
        self, pair: DistinctPair, key: Any, since_us: int, limit: int
    ) -> list[str]:
        self._check()
        seen = self._distinct.get(pair, {}).get(key, {})
        recent = sorted((ts, str(v)) for v, ts in seen.items() if ts > since_us)
        return [v for _, v in reversed(recent)][:limit]  # most recent first


async def replay(store: FeatureStore, rows: list[dict[str, Any]], fs: Any) -> list[dict[str, Any]]:
    """Score-then-update every canonical row in order; return online features per row.

    Each row's label (if any) is handed to the store right after the row: the store only lets it
    count from its visible snapshot, which is what an on-time label emitter achieves.
    """
    from fraudshield.features.online import compute_online

    out = []
    attr_cols = store.spec.te_columns
    for row in rows:
        out.append(compute_online(row, await store.fetch(row), fs))
        await store.update(row)
        if row.get("label") is not None and row.get("label_available_time") is not None:
            attrs = {c: row.get(c) for c in attr_cols}
            await store.add_label(
                row["txn_id"], int(row["label"]), row["label_available_time"], attrs
            )
    return out
