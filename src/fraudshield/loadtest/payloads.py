"""Request bodies for the load test: test-window transactions, 70 % returning cards / 30 % new."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from fraudshield.streaming.replayer import raw_message


def write_payloads(window: pl.DataFrame, path: Path) -> int:
    """Canonical rows -> ``/score`` bodies, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in window.iter_rows(named=True):
            f.write(orjson.dumps(raw_message(row)) + b"\n")
    return window.height


def read_payloads(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


class PayloadPool:
    """Draws bodies at random; each request gets a unique ``txn_id``.

    * a ``known_share`` of requests keep the card (``uid``) of the drawn transaction: returning
      cards whose history builds up as the test runs (after the warm-up stage);
    * the rest get a never-seen card (no history: first-seen flags, empty windows).

    Cards and transaction ids are prefixed with ``run_id`` so runs never share history.
    ``event_time`` is dropped: the API stamps server time, so windows fill in real time.
    """

    def __init__(
        self,
        bodies: Sequence[Mapping[str, Any]],
        *,
        known_share: float = 0.7,
        seed: int = 0,
        run_id: str = "lt",
        worker: str = "0",
    ) -> None:
        if not bodies:
            raise ValueError("empty payload pool")
        if not 0.0 <= known_share <= 1.0:
            raise ValueError(f"known_share must be in [0, 1], got {known_share}")
        self.bodies = list(bodies)
        self.known_share = known_share
        self.rng = random.Random(f"{seed}:{worker}")  # noqa: S311 - load generation, not crypto
        self.prefix = f"{run_id}:{worker}"
        self.run_id = run_id
        self.n = 0
        self.known = 0

    def next(self) -> dict[str, Any]:
        body = dict(self.rng.choice(self.bodies))
        self.n += 1
        body["txn_id"] = f"{self.prefix}:t{self.n}"
        if self.rng.random() < self.known_share:
            body["uid"] = f"{self.run_id}:{body['uid']}"
            self.known += 1
        else:
            body["uid"] = f"{self.prefix}:new{self.n}"
        body.pop("event_time", None)
        return body

    def take(self, n: int) -> Iterable[dict[str, Any]]:
        return (self.next() for _ in range(n))
