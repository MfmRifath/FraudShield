"""Locust scenario for ``POST /score`` (run by ``python -m fraudshield.loadtest run``).

Environment:
  LT_PAYLOADS      JSONL of /score bodies (default artifacts/loadtest/payloads.jsonl)
  LT_KNOWN_SHARE   share of requests from returning cards (default 0.7)
  LT_RUN_ID        prefix for cards and txn ids (runs never share history)
  LT_STAGE         warm-up / scenario name: keeps txn ids unique across the stages of a run
  LT_EXPLAIN       "false" -> POST /score?explain=false (no model reason codes)
  LT_RPS_PER_USER  mean request rate per simulated user (default 2), Poisson arrivals

Besides Locust's own timing of ``/score`` it records the API's reported ``latency_ms`` as a
separate ``service`` series, so the report can separate service time from transport.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from locust import FastHttpUser, poisson, task

from fraudshield.loadtest.payloads import PayloadPool, read_payloads

BODIES = read_payloads(Path(os.environ.get("LT_PAYLOADS", "artifacts/loadtest/payloads.jsonl")))
KNOWN_SHARE = float(os.environ.get("LT_KNOWN_SHARE", "0.7"))
RUN_ID = os.environ.get("LT_RUN_ID", "lt")
STAGE = os.environ.get("LT_STAGE", "run")
PATH = "/score" if os.environ.get("LT_EXPLAIN", "true") != "false" else "/score?explain=false"
_counter = iter(range(1_000_000))


class Scorer(FastHttpUser):
    # Poisson arrivals (exponential gaps), like independent cardholders. Constant pacing with all
    # users spawned together fires in lockstep bursts and measures the load generator instead.
    wait_time = poisson(float(os.environ.get("LT_RPS_PER_USER", "2")))

    def on_start(self) -> None:
        self.pool = PayloadPool(
            BODIES, known_share=KNOWN_SHARE, run_id=RUN_ID, worker=f"{STAGE}-u{next(_counter)}"
        )

    @task
    def score(self) -> None:
        with self.client.post(PATH, json=self.pool.next(), name="/score", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body: dict[str, Any] = r.json()
            r.success()
        self.environment.events.request.fire(
            request_type="API", name="service", response_time=float(body["latency_ms"]),
            response_length=0, exception=None, context={},
        )  # fmt: skip
