"""Parquet sink: decisions -> hourly-partitioned Parquet files (the served-traffic log).

Layout: ``<parquet_dir>/decisions/date=YYYY-MM-DD/hour=HH/part-<n>-<uuid>.parquet`` by event
time (UTC). Buffers are flushed when ``max_rows`` is reached, every ``flush_s`` by a background
timer (so an idle stream still gets written), and on shutdown. Features are stored as JSON text so the schema is stable across
datasets; monitoring (Phase 6) reads them back.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from fraudshield.config import Settings
from fraudshield.streaming.common import WorkerMetrics, json_topic, make_app, parse_time

log = logging.getLogger(__name__)

SCHEMA: dict[str, pl.DataType] = {
    "txn_id": pl.String(),
    "uid": pl.String(),
    "event_time": pl.Datetime("us", "UTC"),
    "amount": pl.Float64(),
    "decision": pl.String(),
    "probability": pl.Float64(),
    "model_version": pl.String(),
    "variant": pl.String(),
    "degraded": pl.Boolean(),
    "latency_ms": pl.Float64(),
    "rules_fired": pl.List(pl.String()),
    "reason_codes": pl.String(),
    "features": pl.String(),
}


def to_record(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "txn_id": event["txn_id"],
        "uid": event["uid"],
        "event_time": parse_time(event["event_time"]),
        "amount": event.get("amount"),
        "decision": event["decision"],
        "probability": event.get("probability"),
        "model_version": event.get("model_version"),
        "variant": event.get("variant"),
        "degraded": event.get("degraded"),
        "latency_ms": event.get("latency_ms"),
        "rules_fired": event.get("rules_fired") or [],
        "reason_codes": orjson.dumps(event.get("reason_codes") or []).decode(),
        "features": orjson.dumps(event.get("features") or {}).decode(),
    }


class HourlyParquetWriter:
    def __init__(
        self,
        base_dir: Path,
        flush_s: float,
        max_rows: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_dir = base_dir / "decisions"
        self.flush_s = flush_s
        self.max_rows = max_rows
        self._clock = clock
        self._buffers: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._last_flush = clock()
        self._part = 0
        self.rows_written = 0
        self._lock = (
            threading.Lock()
        )  # add() runs on the consumer thread, flush() also on the timer
        self._stop = threading.Event()
        self._timer: threading.Thread | None = None

    def add(self, event: dict[str, Any]) -> list[Path]:
        rec = to_record(event)
        t = rec["event_time"]
        with self._lock:
            self._buffers.setdefault((t.strftime("%Y-%m-%d"), t.strftime("%H")), []).append(rec)
            buffered = sum(len(b) for b in self._buffers.values())
            due = self._clock() - self._last_flush >= self.flush_s
        if buffered >= self.max_rows or due:
            return self.flush()
        return []

    def start_timer(self) -> None:
        """Flush every ``flush_s`` even when no messages arrive."""

        def loop() -> None:
            while not self._stop.wait(self.flush_s):
                try:
                    for p in self.flush():
                        log.info("wrote %s", p)
                except Exception:
                    log.exception("periodic parquet flush failed")

        self._timer = threading.Thread(target=loop, name="parquet-flush", daemon=True)
        self._timer.start()

    def close(self) -> list[Path]:
        self._stop.set()
        if self._timer is not None:
            self._timer.join(timeout=5)
        return self.flush()

    def flush(self) -> list[Path]:
        with self._lock:
            buffers, self._buffers = self._buffers, {}
            self._last_flush = self._clock()
        written = []
        for (date, hour), rows in buffers.items():
            if not rows:
                continue
            d = self.base_dir / f"date={date}" / f"hour={hour}"
            d.mkdir(parents=True, exist_ok=True)
            with self._lock:
                part = self._part
                self._part += 1
            path = d / f"part-{part:06d}-{uuid.uuid4().hex[:8]}.parquet"
            pl.DataFrame(rows, schema=SCHEMA).write_parquet(path, compression="zstd")
            with self._lock:
                self.rows_written += len(rows)
            written.append(path)
        return written


def run(settings: Settings) -> None:
    st = settings.streaming
    writer = HourlyParquetWriter(st.parquet_dir, st.parquet_flush_s, st.parquet_max_rows)
    metrics = WorkerMetrics("parquet_sink")
    metrics.serve(st.metrics_port)
    app = make_app(settings, "fraudshield-parquet-sink")

    def handle(value: dict[str, Any], key: Any, ts: int, headers: Any) -> None:
        for p in writer.add(value):
            log.info("wrote %s", p)
        metrics.observe(st.topic_decisions, ts)

    app.dataframe(json_topic(app, settings, st.topic_decisions)).update(handle, metadata=True)
    writer.start_timer()
    try:
        app.run()
    finally:
        writer.close()
