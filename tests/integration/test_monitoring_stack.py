"""Monitoring profile wiring (``make test-integration-all``): targets, rules, dashboards."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.monitoring]


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 - compose-internal
        return json.loads(r.read())


def test_prometheus_scrapes_services_and_loaded_rules() -> None:
    targets = _get("http://prometheus:9090/api/v1/targets")["data"]["activeTargets"]
    up = {t["labels"]["job"] for t in targets if t["health"] == "up"}
    assert {"api", "stream-workers", "prefect-worker", "prometheus"} <= up, up
    rules = {
        r["name"]
        for g in _get("http://prometheus:9090/api/v1/rules")["data"]["groups"]
        for r in g["rules"]
    }
    assert {
        "ScoreLatencyP95High",
        "FallbackRateHigh",
        "ConsumerLagHigh",
        "DriftDetected",
        "ServiceTargetDown",
    } <= rules


def test_grafana_provisions_the_four_dashboards() -> None:
    titles = {d["title"] for d in _get("http://grafana:3000/api/search?tag=fraudshield")}
    assert titles == {
        "FraudShield / Service SLOs",
        "FraudShield / Decision & Score mix",
        "FraudShield / Champion vs Challenger",
        "FraudShield / Pipeline health",
    }
    ds = _get("http://grafana:3000/api/datasources/uid/prometheus")  # anonymous viewer can read
    assert ds["url"] == "http://prometheus:9090"
