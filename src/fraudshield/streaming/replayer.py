"""Replay canonical transactions into Kafka in event-time order.

* ``transactions.raw``: the /score request body (no label fields), key = uid.
* ``transactions.truth``: ground truth for labelled rows (txn_id, uid, event_time, label, attrs),
  read only by the label emitter, which releases each label at its availability time.

Pacing: ``speed`` = event-time seconds per wall second (0 = no event-time pacing); ``rate`` =
max events per wall second (0 = unlimited). ``namespace`` prefixes ids and entity values so
repeated runs (e.g. integration tests) do not mix with state from earlier runs. ``drift_share``
> 0 turns that share of events into a "fraud ring" (``inject_fraud_ring``) for drift demos.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import orjson
import polars as pl

from fraudshield.adapters import get_adapter
from fraudshield.config import DatasetName, Settings
from fraudshield.labels import unit_hash
from fraudshield.pipeline import DatasetPaths
from fraudshield.schema import EXTRA_PREFIX

log = logging.getLogger(__name__)
FIXTURE = Path("tests/fixtures/sparkov_sample.csv")
_REQUEST_FIELDS = (
    "txn_id", "uid", "amount", "product", "card_network", "card_kind", "p_email_domain",
    "r_email_domain", "device_type", "device_info", "addr",
)  # fmt: skip
NAMESPACED = ("txn_id", "uid", "device_info", "p_email_domain", "addr")


class Producer(Protocol):
    def produce(self, topic: str, value: bytes, key: bytes) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float) -> int: ...


def load_source(
    settings: Settings, source: str, dataset: DatasetName, path: Path | None = None
) -> pl.DataFrame:
    """``fixture``: the committed CC0 Sparkov sample; ``canonical``: data/interim/<ds>/transactions.parquet."""
    if source == "fixture":
        return get_adapter("sparkov", settings).load([path or FIXTURE])
    if source == "canonical":
        return pl.read_parquet(path or DatasetPaths.of(settings, dataset).canonical)
    raise ValueError(f"unknown source {source}")


def select_window(
    df: pl.DataFrame, start: datetime | None, end: datetime | None, max_events: int | None
) -> pl.DataFrame:
    t = pl.col("event_time")
    if start is not None:
        df = df.filter(t >= start)
    if end is not None:
        df = df.filter(t < end)
    df = df.sort("event_time")
    return df.head(max_events) if max_events else df


def namespaced(row: Mapping[str, Any], ns: str | None) -> dict[str, Any]:
    out = dict(row)
    if ns:
        for k in NAMESPACED:
            if out.get(k) is not None:
                out[k] = f"{ns}:{out[k]}"
    return out


def raw_message(row: Mapping[str, Any]) -> dict[str, Any]:
    msg = {k: row.get(k) for k in _REQUEST_FIELDS}
    msg["event_time"] = row["event_time"].isoformat()
    msg["extra"] = {
        k.removeprefix(EXTRA_PREFIX): v for k, v in row.items() if k.startswith(EXTRA_PREFIX)
    }
    return msg


def truth_message(row: Mapping[str, Any], attr_columns: Iterable[str]) -> dict[str, Any] | None:
    if row.get("label") is None:
        return None
    return {
        "txn_id": row["txn_id"],
        "uid": row["uid"],
        "event_time": row["event_time"].isoformat(),
        "label": int(row["label"]),
        "attrs": {c: row.get(c) for c in attr_columns},
    }


def inject_fraud_ring(row: Mapping[str, Any], share: float, seed: int = 7) -> dict[str, Any]:
    """Drift injection ("fraud ring"): a share of events move onto a few shared devices and one
    email domain, with larger amounts, and are fraud. Deterministic per txn_id.
    """
    out = dict(row)
    if unit_hash(str(row["txn_id"]), seed) >= share:
        return out
    k = int(unit_hash(str(row["txn_id"]), seed + 1) * 3)
    out.update(device_info=f"ring-device-{k}", device_type="mobile", p_email_domain="ring-mail.example",
               amount=round(float(row["amount"]) * 4 + 50, 2), label=1)  # fmt: skip
    return out


def pacing_delay(prev: datetime | None, cur: datetime, speed: float, rate: float) -> float:
    """Wall seconds to wait before sending ``cur``."""
    wait = 0.0
    if speed > 0 and prev is not None:
        wait = max(wait, (cur - prev).total_seconds() / speed)
    if rate > 0:
        wait = max(wait, 1.0 / rate)
    return wait


@dataclass
class ReplayStats:
    raw: int = 0
    truth: int = 0


def replay(
    rows: Iterable[Mapping[str, Any]],
    producer: Producer,
    settings: Settings,
    *,
    speed: float = 0.0,
    rate: float = 0.0,
    namespace: str | None = None,
    drift_share: float = 0.0,
    attr_columns: Iterable[str] = ("p_email_domain", "device_info", "addr"),
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayStats:
    st = settings.streaming
    attrs = tuple(attr_columns)
    stats = ReplayStats()
    prev: datetime | None = None
    for row in rows:
        r = namespaced(inject_fraud_ring(row, drift_share) if drift_share > 0 else row, namespace)
        wait = pacing_delay(prev, r["event_time"], speed, rate)
        if wait > 0:
            sleep(wait)
        prev = r["event_time"]
        key = str(r["uid"]).encode()
        producer.produce(st.topic_raw, value=orjson.dumps(raw_message(r)), key=key)
        stats.raw += 1
        truth = truth_message(r, attrs)
        if truth is not None:
            producer.produce(st.topic_truth, value=orjson.dumps(truth), key=key)
            stats.truth += 1
        producer.poll(0)
    producer.flush(30)
    log.info("replayed %d transactions (%d with truth)", stats.raw, stats.truth)
    return stats
