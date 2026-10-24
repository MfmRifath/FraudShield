"""Prometheus metrics (one registry per app instance, so tests can build many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
# every reason ServingService can report (scorer.py); zero-initialised for dashboards
FALLBACK_REASONS = (
    "feature_timeout", "feature_store_error", "circuit_open", "model_error", "no_model",
    "internal_error", "review_capacity",
)  # fmt: skip
SCORE_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


class ServingMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.score_latency = Histogram(
            "score_latency_seconds", "In-process /score latency", ["model_version", "degraded"],
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.decisions = Counter(
            "decisions_total", "Decisions by outcome", ["decision"], registry=r
        )
        self.fraud_score = Histogram(
            "fraud_score", "Calibrated fraud probability", buckets=SCORE_BUCKETS, registry=r
        )
        self.fallbacks = Counter("fallback_total", "Fallback paths taken", ["reason"], registry=r)
        self.feature_fetch = Histogram(
            "feature_fetch_seconds",
            "Feature-store fetch latency",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.review_queue_depth = Gauge("review_queue_depth", "Open review cases", registry=r)
        self.model_version_info = Gauge(
            "model_version_info",
            "Loaded model versions (value 1)",
            ["alias", "version"],
            registry=r,
        )
        self.publish_dropped = Counter(
            "publish_dropped_total",
            "Events dropped because the publish queue was full",
            ["topic"],
            registry=r,
        )

        # zero-initialise known label sets so dashboards show 0 instead of "No data"
        for topic in ("decisions", "labels"):
            self.publish_dropped.labels(topic=topic)
        for reason in FALLBACK_REASONS:
            self.fallbacks.labels(reason=reason)

    def set_model_versions(self, versions: dict[str, str | None]) -> None:
        self.model_version_info.clear()
        for alias, version in versions.items():
            if version is not None:
                self.model_version_info.labels(alias=alias, version=version).set(1)
