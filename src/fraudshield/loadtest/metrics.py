"""Snapshots of the API's Prometheus metrics, and differences between them."""

from __future__ import annotations

import math
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field

from prometheus_client.parser import text_string_to_metric_families

from fraudshield.monitoring.summary import bucket_quantiles


@dataclass(frozen=True)
class Snapshot:
    t: float
    decisions: Mapping[str, float] = field(default_factory=dict)
    fallbacks: Mapping[str, float] = field(default_factory=dict)
    latency_buckets: Mapping[float, float] = field(default_factory=dict)  # le -> cumulative count

    @property
    def scored(self) -> float:
        return sum(self.decisions.values())

    def minus(self, earlier: Snapshot) -> Snapshot:
        def sub(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
            return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in sorted({*a, *b})}

        lat = {
            le: c - earlier.latency_buckets.get(le, 0.0) for le, c in self.latency_buckets.items()
        }
        return Snapshot(self.t - earlier.t, sub(self.decisions, earlier.decisions),
                        sub(self.fallbacks, earlier.fallbacks), lat)  # fmt: skip

    def latency_quantiles(self) -> dict[str, float | None]:
        return bucket_quantiles(self.latency_buckets)


def parse_metrics(text: str, t: float) -> Snapshot:
    """Decision and fallback counters plus score-latency buckets (summed over label sets)."""
    decisions: dict[str, float] = {}
    fallbacks: dict[str, float] = {}
    buckets: dict[float, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name == "decisions_total":
                k = s.labels["decision"]
                decisions[k] = decisions.get(k, 0.0) + s.value
            elif s.name == "fallback_total":
                k = s.labels["reason"]
                fallbacks[k] = fallbacks.get(k, 0.0) + s.value
            elif s.name == "score_latency_seconds_bucket":
                le = math.inf if s.labels["le"] == "+Inf" else float(s.labels["le"])
                buckets[le] = buckets.get(le, 0.0) + s.value
    return Snapshot(t, decisions, fallbacks, buckets)


def scrape(url: str, t: float, timeout: float = 5.0) -> Snapshot:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - operator-given URL
        return parse_metrics(r.read().decode(), t)
