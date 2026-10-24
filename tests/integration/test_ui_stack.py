"""UI profile (``make test-integration-all``): nginx serves the UI and proxies /api, SSE unbuffered.

Run with the mlops profile up too, so the API's case store is Postgres (``FS_CASE_STORE``).
"""

from __future__ import annotations

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import psycopg
import pytest

from fraudshield.config import Settings
from fraudshield.streaming.replayer import load_source
from tests.serving_support import payload

pytestmark = [pytest.mark.integration, pytest.mark.ui, pytest.mark.mlops]
UI = "http://ui:8080"
PG = "postgresql://fraudshield:fraudshield@postgres:5432/fraudshield"


def _get(path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"{UI}{path}", timeout=10) as r:  # noqa: S310 - compose-internal
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        with e:  # an HTTPError holds the response (and its socket) open
            return e.code, e.read()


def _score_body() -> dict[str, Any]:
    row = load_source(Settings(), "fixture", "sparkov").row(0, named=True)
    return payload(row) | {"txn_id": f"ui-it-{uuid.uuid4().hex[:10]}"}


def test_ui_is_served_and_spa_routes_fall_back_to_index() -> None:
    for path in ("/", "/queue", "/cases/some-id"):
        status, body = _get(path)
        assert status == 200, path
        assert b'<div id="root">' in body


def test_api_is_proxied_and_admin_is_not() -> None:
    status, body = _get("/api/health")
    assert (status, json.loads(body)) == (200, {"status": "ok"})
    assert _get("/api/cases?limit=1")[0] == 200
    assert _get("/api/monitoring/summary")[0] == 200
    assert _get("/api/metrics")[0] == 403
    req = urllib.request.Request(f"{UI}/api/admin/reload", method="POST")  # noqa: S310
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
    e.value.close()
    assert e.value.code == 403


def test_decision_stream_arrives_through_the_proxy_unbuffered() -> None:
    with contextlib.ExitStack() as stack:
        conn = stack.enter_context(
            contextlib.closing(http.client.HTTPConnection("ui", 8080, timeout=15))
        )
        conn.request("GET", "/api/events/stream", headers={"Accept": "text/event-stream"})
        resp = stack.enter_context(contextlib.closing(conn.getresponse()))
        assert resp.status == 200
        assert resp.getheader("content-type", "").startswith("text/event-stream")
        assert resp.readline().startswith(b":")  # the opening comment is flushed immediately
        body = _score_body()
        headers = {"content-type": "application/json"}
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"{UI}/api/score", data=data, headers=headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            assert r.status == 200
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = resp.readline()
            if line.startswith(b"data:") and body["txn_id"].encode() in line:
                break
        else:
            pytest.fail("decision event did not arrive through nginx within 15 s")


def test_api_case_store_is_postgres() -> None:
    with psycopg.Connection.connect(PG) as conn:
        row = conn.execute(
            "SELECT to_regclass('public.cases'), to_regclass('public.labels'), "
            "to_regclass('public.audit_log')"
        ).fetchone()
    assert row is not None
    assert all(row), row  # created by the API at startup
