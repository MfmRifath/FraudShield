"""API contract tests (TestClient runs the real lifespan)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraudshield.features.definitions import WindowAgg
from fraudshield.features.offline import compute_offline
from fraudshield.serving.app import create_app
from fraudshield.serving.events import MemorySink
from fraudshield.serving.models import canary_bucket
from fraudshield.synthetic import synthetic_canonical
from tests.serving_support import (
    ALWAYS_REVIEW_RULES,
    TOKEN,
    make_bundle,
    payload,
    serving_settings,
    window_rows,
    write_yaml,
)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("bundle")
    return make_bundle(serving_settings(tmp), tmp / "bundle")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return window_rows()


@pytest.fixture
def client(bundle_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    s = serving_settings(tmp_path, bundle_dir=bundle_dir)
    with TestClient(create_app(s)) as c:
        yield c


@pytest.mark.unit
def test_health_ready_model(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["models"]["champion"] == "synthetic-20x15"
    assert client.get("/model").json()["champion"] == "synthetic-20x15"


@pytest.mark.unit
def test_score_contract(client: TestClient, rows: list[dict[str, Any]]) -> None:
    r = client.post("/score", json=payload(rows[0]))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"txn_id", "probability", "decision", "reason_codes", "model_version", "degraded",
                         "latency_ms", "variant", "rules_fired", "fallback"}  # fmt: skip
    assert body["txn_id"] == rows[0]["txn_id"]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "block"}
    assert len(body["reason_codes"]) <= 3
    assert body["model_version"] == "synthetic-20x15"
    assert body["degraded"] is False
    assert body["variant"] == "champion"
    assert body["latency_ms"] > 0
    assert body["fallback"] is None


@pytest.mark.unit
def test_score_without_explanations(
    bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    """``explain=false`` (load-test scenario): same probability, only rule reason codes."""
    s = serving_settings(tmp_path, bundle_dir=bundle_dir)
    with TestClient(create_app(s)) as full_c, TestClient(create_app(s)) as lean_c:
        for row in rows[:30]:  # same sequence into two fresh apps -> same feature history
            full = full_c.post("/score", json=payload(row)).json()
            lean = lean_c.post("/score", params={"explain": "false"}, json=payload(row)).json()
            assert lean["probability"] == pytest.approx(full["probability"], abs=1e-12)
            assert lean["decision"] == full["decision"]
            assert all(c["code"].startswith("RULE:") for c in lean["reason_codes"])
            assert any(not c["code"].startswith("RULE:") for c in full["reason_codes"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: b.pop("uid"), id="missing-uid"),
        pytest.param(lambda b: b.update(amount=-1), id="negative-amount"),
        pytest.param(lambda b: b.update(event_time="2021-03-01T10:00:00"), id="naive-time"),
        pytest.param(lambda b: b.update(unexpected=1), id="unknown-field"),
        pytest.param(lambda b: b.update(txn_id=""), id="empty-txn-id"),
    ],
)
def test_invalid_payloads_are_422(
    client: TestClient, rows: list[dict[str, Any]], mutate: Any
) -> None:
    body = payload(rows[0])
    mutate(body)
    assert client.post("/score", json=body).status_code == 422


@pytest.mark.unit
def test_openapi_documents_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["ScoreResponse"]["properties"]
    assert {
        "probability",
        "decision",
        "reason_codes",
        "model_version",
        "degraded",
        "latency_ms",
    } <= set(props)


@pytest.mark.unit
def test_metrics_exposed(client: TestClient, rows: list[dict[str, Any]]) -> None:
    for row in rows[:5]:
        client.post("/score", json=payload(row))
    text = client.get("/metrics").text
    for name in ("score_latency_seconds_bucket", "decisions_total", "fraud_score_bucket", "feature_fetch_seconds",
                 "review_queue_depth", 'model_version_info{alias="champion",version="synthetic-20x15"}'):  # fmt: skip
        assert name in text, name


@pytest.mark.unit
def test_api_features_match_offline_on_replayed_prefix(bundle_dir: Path, tmp_path: Path) -> None:
    """End to end: replay 1,500 events through HTTP; published features == offline features.

    Target encodings are excluded (no labels flow through /score); everything else must match.
    """
    s = serving_settings(tmp_path, bundle_dir=bundle_dir)
    canon = synthetic_canonical(10_000, seed=0).head(1500)
    sink = MemorySink()
    with TestClient(create_app(s, sink=sink)) as c:
        for row in canon.to_dicts():
            assert c.post("/score", json=payload(row)).status_code == 200
    published = {m["txn_id"]: m["features"] for m in sink.decoded("decisions")}
    from fraudshield.pipeline import feature_set_for

    fs = feature_set_for(s, "sparkov")
    offline = compute_offline(canon, fs).to_dicts()
    names = [n for n in fs.names if not n.startswith("te_")]
    assert any(isinstance(f, WindowAgg) for f in fs.features)
    worst = 0.0
    for off in offline:
        on = published[off["txn_id"]]
        for n in names:
            a, b = on[n], off[n]
            if a is None or b is None:
                assert (a, b) == (None, None), (off["txn_id"], n, a, b)
            elif isinstance(b, str):
                assert a == b
            else:
                worst = max(worst, abs(float(a) - float(b)))
    assert worst < 1e-6


@pytest.mark.unit
def test_review_cases_and_labels(
    bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    rules = write_yaml(tmp_path / "rules.yaml", ALWAYS_REVIEW_RULES)
    s = serving_settings(tmp_path, bundle_dir=bundle_dir, rules=rules)
    sink = MemorySink()
    with TestClient(create_app(s, sink=sink)) as c:
        for row in rows[:20]:
            body = c.post("/score", json=payload(row)).json()
            assert body["decision"] in {"review", "block"}
            assert body["rules_fired"] == ["review_everything"]
            assert body["reason_codes"][0]["code"] == "RULE:review_everything"
        cases = c.get("/cases").json()
        losses = [x["expected_loss"] for x in cases]
        assert losses == sorted(losses, reverse=True)
        target = cases[0]["case_id"]
        assert c.get(f"/cases/{target}").json()["status"] == "open"
        labeled = c.post(
            f"/cases/{target}/label", json={"label": "fraud", "analyst_id": "a1", "notes": "ring"}
        )
        assert labeled.status_code == 200
        assert labeled.json()["status"] == "labeled"
        assert labeled.json()["label_source"] == "analyst"
        assert target not in {x["case_id"] for x in c.get("/cases").json()}
        assert target in {x["case_id"] for x in c.get("/cases?status=labeled").json()}
        assert c.get("/cases/nope").status_code == 404
        assert (
            c.post("/cases/nope/label", json={"label": "fraud", "analyst_id": "a1"}).status_code
            == 404
        )
        assert (
            c.post(
                f"/cases/{target}/label", json={"label": "maybe", "analyst_id": "a1"}
            ).status_code
            == 422
        )
        depth = [
            ln for ln in c.get("/metrics").text.splitlines() if ln.startswith("review_queue_depth")
        ]
        assert depth == [f"review_queue_depth {len(cases) - 1}.0"]
    labels = sink.decoded("labels")
    assert labels[0]["txn_id"] == target
    assert labels[0]["label"] == 1
    assert labels[0]["label_source"] == "analyst"


@pytest.mark.unit
def test_admin_reload(bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    live = tmp_path / "live_bundle"
    shutil.copytree(bundle_dir, live)
    s = serving_settings(tmp_path, bundle_dir=live)
    with TestClient(create_app(s)) as c:
        assert c.post("/admin/reload").status_code == 401
        assert c.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        ok = c.post("/admin/reload", headers={"X-Admin-Token": TOKEN})
        assert ok.status_code == 200
        assert ok.json()["models"]["champion"] == "synthetic-20x15"
        shutil.rmtree(live)  # a broken reload must keep serving the previous model
        bad = c.post("/admin/reload", headers={"X-Admin-Token": TOKEN})
        assert bad.status_code == 409
        assert c.post("/score", json=payload(rows[0])).json()["model_version"] == "synthetic-20x15"
    no_token = serving_settings(tmp_path, bundle_dir=bundle_dir, admin_token=None)
    with TestClient(create_app(no_token)) as c:
        assert c.post("/admin/reload", headers={"X-Admin-Token": TOKEN}).status_code == 403


@pytest.mark.unit
def test_canary_routing_is_stable(
    bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    s0 = serving_settings(tmp_path, bundle_dir=bundle_dir)
    challenger = make_bundle(s0, tmp_path / "challenger", rounds=10)
    s = serving_settings(
        tmp_path, bundle_dir=bundle_dir, challenger_bundle_dir=challenger, canary_pct=50
    )
    with TestClient(create_app(s)) as c:
        seen: dict[str, set[str]] = {}
        for row in rows[:300]:
            body = c.post("/score", json=payload(row)).json()
            seen.setdefault(row["uid"], set()).add(body["variant"])
            want = "challenger" if canary_bucket(row["uid"]) < 50 else "champion"
            assert body["variant"] == want
            assert body["model_version"] == (
                "synthetic-10x15" if want == "challenger" else "synthetic-20x15"
            )
        assert all(len(v) == 1 for v in seen.values())
        assert {"champion", "challenger"} == set().union(*seen.values())
    assert canary_bucket("u001") == canary_bucket("u001")


@pytest.mark.unit
def test_sse_stream_opens(client: TestClient) -> None:
    with client.stream("GET", "/events/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        first = next(r.iter_lines())
        assert first == ": fraudshield decisions"


@pytest.mark.unit
def test_model_threads_match_inline_scoring(
    bundle_dir: Path, tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    """The executor path (default) returns exactly what inline scoring returns, also concurrently."""
    import asyncio

    from fraudshield.serving.schemas import ScoreRequest

    base = serving_settings(tmp_path, bundle_dir=bundle_dir, update_store_inline=False)
    inline = base.model_copy(
        update={"serving": base.serving.model_copy(update={"model_threads": 0})}
    )
    reqs = [ScoreRequest.model_validate(payload(r)) for r in rows[:60]]
    with TestClient(create_app(base)) as c, TestClient(create_app(inline)) as ci:
        svc, svc_inline = c.app.state.fs.service, ci.app.state.fs.service  # type: ignore[attr-defined]
        assert svc.model_executor is not None
        assert svc_inline.model_executor is None

        async def concurrent() -> list[Any]:
            return list(await asyncio.gather(*(svc.score(r) for r in reqs)))

        threaded = c.portal.call(concurrent)  # type: ignore[union-attr]
        sequential = [ci.portal.call(svc_inline.score, r) for r in reqs]  # type: ignore[union-attr]
    for a, b in zip(threaded, sequential, strict=True):
        assert (a.probability, a.decision) == (b.probability, b.decision)
        assert [x.code for x in a.reason_codes] == [x.code for x in b.reason_codes]
