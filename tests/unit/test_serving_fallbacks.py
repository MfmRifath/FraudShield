"""Fallback behaviour: the service degrades, it never returns 5xx for internal faults."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraudshield.features.online import EntityState, OnlineSpec
from fraudshield.features.store import FeatureStoreError, InMemoryFeatureStore
from fraudshield.serving.app import create_app
from fraudshield.serving.breaker import CircuitBreaker
from fraudshield.serving.events import Broadcaster
from tests.serving_support import (
    ALWAYS_REVIEW_RULES,
    make_bundle,
    payload,
    serving_settings,
    window_rows,
    write_yaml,
)


class FlakyStore(InMemoryFeatureStore):
    """In-memory store whose fetch can fail or hang on demand; counts calls."""

    def __init__(self, spec: OnlineSpec) -> None:
        super().__init__(spec)
        self.mode = "ok"
        self.fetches = 0

    async def fetch(self, row: Any) -> EntityState:
        self.fetches += 1
        if self.mode == "error":
            raise FeatureStoreError("redis down")
        if self.mode == "hang":
            await asyncio.sleep(0.2)
        return await super().fetch(row)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("bundle")
    return make_bundle(serving_settings(tmp), tmp / "bundle")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return window_rows()


def _app(tmp: Path, bundle: Path, **kw: Any) -> tuple[Any, FlakyStore]:
    s = serving_settings(tmp, bundle_dir=bundle, **kw)
    from fraudshield.pipeline import feature_set_for

    store = FlakyStore(OnlineSpec.from_feature_set(feature_set_for(s, "sparkov")))
    return create_app(s, store=store), store


@pytest.fixture
def flaky(bundle_dir: Path, tmp_path: Path) -> Iterator[tuple[TestClient, FlakyStore]]:
    app, store = _app(tmp_path, bundle_dir, breaker_failure_threshold=3)
    with TestClient(app) as c:
        yield c, store


def _fallbacks(c: TestClient) -> dict[str, float]:
    out = {}
    for ln in c.get("/metrics").text.splitlines():
        if ln.startswith("fallback_total{"):
            reason = ln.split('reason="')[1].split('"')[0]
            out[reason] = float(ln.rsplit(" ", 1)[1])
    return out


@pytest.mark.unit
def test_store_error_uses_degraded_model(
    flaky: tuple[TestClient, FlakyStore], rows: list[dict[str, Any]]
) -> None:
    c, store = flaky
    store.mode = "error"
    body = c.post("/score", json=payload(rows[0])).json()
    assert body["degraded"] is True
    assert body["variant"] == "champion"
    assert body["fallback"] == "feature_store_error"
    assert body["probability"] is not None
    from fraudshield.features.definitions import build_feature_set
    from fraudshield.synthetic import EXTRAS

    request_only = set(build_feature_set(EXTRAS, 50.0).request_only)
    assert all(r["feature"] in request_only for r in body["reason_codes"] if r["feature"])
    assert _fallbacks(c)["feature_store_error"] == 1.0


@pytest.mark.unit
def test_timeout_and_circuit_breaker(
    flaky: tuple[TestClient, FlakyStore], rows: list[dict[str, Any]]
) -> None:
    c, store = flaky
    store.mode = "hang"
    first = c.post("/score", json=payload(rows[0])).json()
    assert first["fallback"] == "feature_timeout"
    assert first["degraded"] is True
    assert first["latency_ms"] < 150  # 10 ms budget, not the 200 ms hang
    for row in rows[1:3]:
        c.post("/score", json=payload(row))
    assert store.fetches == 3  # threshold reached -> breaker opens
    opened = c.post("/score", json=payload(rows[3])).json()
    assert opened["fallback"] == "circuit_open"
    assert store.fetches == 3  # the store is no longer called
    assert {"feature_timeout", "circuit_open"} <= set(_fallbacks(c))


def _state(b: CircuitBreaker) -> str:
    return b.state  # fresh read each time (mypy would otherwise keep a narrowed literal)


@pytest.mark.unit
def test_breaker_state_machine() -> None:
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, reset_s=10, clock=lambda: now[0])
    assert b.allow()
    assert _state(b) == "closed"
    b.failure()
    assert _state(b) == "closed"
    b.failure()
    assert _state(b) == "open"
    assert not b.allow()
    now[0] = 10.0
    assert _state(b) == "half_open"
    assert b.allow()  # one trial
    assert not b.allow()  # only one trial at a time
    b.failure()
    assert _state(b) == "open"  # failed trial re-opens for another period
    now[0] = 20.0
    assert b.allow()
    b.success()
    assert _state(b) == "closed"
    assert b.allow()


@pytest.mark.unit
def test_model_error_falls_back_to_rules(
    flaky: tuple[TestClient, FlakyStore], rows: list[dict[str, Any]]
) -> None:
    c, _ = flaky
    slot = c.app.state.fs.models.champion  # type: ignore[attr-defined]

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("corrupt booster")

    slot.bundle.score_one = boom
    r = c.post("/score", json=payload(rows[0]))
    assert r.status_code == 200
    body = r.json()
    assert body["variant"] == "rules_only"
    assert body["probability"] is None
    assert body["model_version"] is None
    assert body["degraded"] is True
    assert body["fallback"] == "model_error"


@pytest.mark.unit
def test_store_down_and_model_error_still_answers(
    flaky: tuple[TestClient, FlakyStore], rows: list[dict[str, Any]]
) -> None:
    c, store = flaky
    store.mode = "error"
    c.app.state.fs.models.champion.bundle.score_one = lambda *a, **k: 1 / 0  # type: ignore[attr-defined]
    r = c.post("/score", json=payload(rows[0]))
    assert r.status_code == 200
    assert r.json()["variant"] == "rules_only"


@pytest.mark.unit
def test_internal_error_is_not_5xx(
    flaky: tuple[TestClient, FlakyStore], rows: list[dict[str, Any]]
) -> None:
    c, _ = flaky
    svc = c.app.state.fs.service  # type: ignore[attr-defined]

    def explode(row: Any) -> Any:
        raise ValueError("bug")

    svc.rules = type("R", (), {"evaluate_row": staticmethod(explode)})()
    r = c.post("/score", json=payload(rows[0]))
    assert r.status_code == 200
    assert r.json()["fallback"] == "internal_error"
    assert r.json()["decision"] == "review"


@pytest.mark.unit
def test_missing_model_not_ready_but_scores_rules_only(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    app, _ = _app(tmp_path, tmp_path / "does-not-exist")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503
        assert "no model bundle" in ready.json()["error"]
        body = c.post("/score", json=payload(rows[0])).json()
        assert body["variant"] == "rules_only"
        assert body["fallback"] == "no_model"


@pytest.mark.unit
def test_review_capacity_per_day(
    bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    rules = write_yaml(tmp_path / "rules.yaml", ALWAYS_REVIEW_RULES)
    costs = write_yaml(
        tmp_path / "costs.yaml",
        "chargeback_fee: 25\nmargin_rate: 0.03\nchurn_penalty: 10\nanalyst_cost: 4\n"
        "review_friction: 2\nreview_capacity_per_day: 2\n",
    )
    app, _ = _app(tmp_path, bundle_dir, rules=rules, costs=costs)
    same_day = [r for r in rows if r["event_time"].date() == rows[0]["event_time"].date()][:6]
    with TestClient(app) as c:
        decisions = [c.post("/score", json=payload(r)).json()["decision"] for r in same_day]
    # block stays block; of the rest, the first 2 take the daily capacity and the
    # rule-forced reviews after that fall back to approve (as in the rules-only baseline)
    rest = [d for d in decisions if d != "block"]
    assert len(rest) >= 3
    assert rest[:2] == ["review", "review"]
    assert set(rest[2:]) == {"approve"}


@pytest.mark.unit
def test_broadcaster_drops_oldest_for_slow_clients() -> None:
    b = Broadcaster(maxsize=2)
    q = b.subscribe()
    for i in range(5):
        b.publish({"i": i})
    assert [q.get_nowait()["i"], q.get_nowait()["i"]] == [3, 4]
    b.unsubscribe(q)
    assert b.n_subscribers == 0


@pytest.mark.unit
def test_store_is_pinged_at_startup(bundle_dir: Path, tmp_path: Path) -> None:
    """Regression: the pool must be warm before the first request (cold Redis connect > 10 ms)."""
    app, store = _app(tmp_path, bundle_dir)
    calls: list[str] = []
    original = store.ping

    async def ping() -> bool:
        calls.append("ping")
        return await original()

    store.ping = ping  # type: ignore[method-assign]
    with TestClient(app):
        assert calls == ["ping"]


@pytest.mark.unit
def test_fallback_reasons_are_all_declared() -> None:
    """Every fallback reason the scorer can emit is zero-initialised for the dashboards."""
    import re

    from fraudshield.serving.metrics import FALLBACK_REASONS

    src = (Path(__file__).resolve().parents[2] / "src/fraudshield/serving/scorer.py").read_text()
    emitted = set(re.findall(r'(?:fallback|reason)\s*=\s*"([a-z_]+)"', src)) | set(
        re.findall(r'return None, "([a-z_]+)"', src)
    )
    assert emitted - {"internal_error"} <= set(FALLBACK_REASONS), emitted - set(FALLBACK_REASONS)
    assert "internal_error" in FALLBACK_REASONS
