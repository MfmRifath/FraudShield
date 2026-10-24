"""Gateway: plays the payment gateway, calling ``POST /score`` for each raw transaction.

Consumes ``transactions.raw`` partition by partition (keyed by uid), so a card's transactions
are scored in order. Transient failures (connection errors, 5xx) are retried with backoff.
4xx responses are logged and skipped; they would fail again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

from fraudshield.config import Settings
from fraudshield.streaming.common import WorkerMetrics, json_topic, make_app

log = logging.getLogger(__name__)


class Gateway:
    def __init__(
        self,
        api_url: str,
        timeout_s: float,
        session: requests.Session | None = None,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = api_url.rstrip("/") + "/score"
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.retries = retries
        self.sleep = sleep

    def handle(self, txn: dict[str, Any]) -> str:
        """Returns the outcome label: ok | client_error | failed."""
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(self.url, json=txn, timeout=self.timeout_s)
            except requests.RequestException as e:
                log.warning("score %s attempt %d failed: %s", txn.get("txn_id"), attempt + 1, e)
            else:
                if r.status_code == 200:
                    return "ok"
                if 400 <= r.status_code < 500:
                    log.error(
                        "score %s rejected (%d): %s", txn.get("txn_id"), r.status_code, r.text[:200]
                    )
                    return "client_error"
                log.warning(
                    "score %s got %d (attempt %d)", txn.get("txn_id"), r.status_code, attempt + 1
                )
            if attempt < self.retries:
                self.sleep(0.2 * 2**attempt)
        return "failed"


def run(settings: Settings) -> None:
    st = settings.streaming
    gw = Gateway(st.api_url, st.gateway_timeout_s)
    metrics = WorkerMetrics("gateway")
    metrics.serve(st.metrics_port)
    app = make_app(settings, "fraudshield-gateway")

    def handle(value: dict[str, Any], key: Any, ts: int, headers: Any) -> None:
        metrics.observe(st.topic_raw, ts, gw.handle(value))

    app.dataframe(json_topic(app, settings, st.topic_raw)).update(handle, metadata=True)
    app.run()
