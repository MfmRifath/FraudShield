"""Single-container demo: in-process replay, mounted API + static UI, bounded memory."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from fraudshield.cases.store import InMemoryCaseStore
from fraudshield.config import Settings
from fraudshield.demo import create_demo_app, load_window, replay
from fraudshield.synthetic import synthetic_canonical
from tests.case_contract import make_case
from tests.serving_support import ALWAYS_REVIEW_RULES, make_bundle, serving_settings, write_yaml


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("demo")
    return make_bundle(serving_settings(tmp), tmp / "bundle")


def _settings(tmp: Path, bundle: Path, n_rows: int = 60, **demo: object) -> Settings:
    window = tmp / "window.parquet"
    synthetic_canonical(10_000, seed=3).head(n_rows).write_parquet(window)
    ui = tmp / "ui"
    (ui / "assets").mkdir(parents=True, exist_ok=True)
    (ui / "index.html").write_text("<div id=root></div>")
    (ui / "assets" / "app.js").write_text("console.log('ui')")
    rules = write_yaml(tmp / "rules.yaml", ALWAYS_REVIEW_RULES)  # every event opens a case
    s = serving_settings(tmp, bundle_dir=bundle, rules=rules, publisher="none")
    cfg = s.demo.model_copy(update={"window": window, "ui_dir": ui, "rate": 1000.0} | demo)
    return s.model_copy(update={"demo": cfg})


@pytest.mark.unit
def test_replay_loops_with_shifted_times_and_unique_ids(bundle_dir: Path, tmp_path: Path) -> None:
    s = _settings(tmp_path, bundle_dir, n_rows=40)
    app = create_demo_app(s)
    api = app.routes[0].app  # type: ignore[attr-defined]  # the mounted API
    with TestClient(api) as c:  # API lifespan only: drive the replayer by hand
        n = c.portal.call(lambda: replay(api, s, max_events=100, pace=False))  # type: ignore[union-attr]
        summary = c.get("/monitoring/summary").json()
        cases = c.get("/cases", params={"limit": 500}).json()
    assert n == 100
    assert sum(summary["decisions"].values()) == 100
    ids = [x["case_id"] for x in cases]
    assert len(ids) == len(set(ids))
    assert {i.rsplit("~", 1)[1] for i in ids if "~" in i} == {"1", "2"}  # 40-row window, 100 events
    end = load_window(s)["event_time"].max()
    assert isinstance(end, datetime)
    later = [datetime.fromisoformat(x["txn"]["event_time"]) for x in cases if "~" in x["case_id"]]
    assert later
    assert all(t > end for t in later)  # looped events are shifted past the window


@pytest.mark.unit
def test_replay_without_loop_stops_at_window_end(bundle_dir: Path, tmp_path: Path) -> None:
    s = _settings(tmp_path, bundle_dir, n_rows=25, loop=False)
    app = create_demo_app(s)
    api = app.routes[0].app  # type: ignore[attr-defined]
    with TestClient(api) as c:
        assert c.portal.call(lambda: replay(api, s, pace=False)) == 25  # type: ignore[union-attr]


@pytest.mark.unit
def test_demo_app_serves_api_ui_and_replays_in_background(bundle_dir: Path, tmp_path: Path) -> None:
    s = _settings(tmp_path, bundle_dir, n_rows=30, loop=False)
    with TestClient(create_demo_app(s)) as c:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if sum(c.get("/api/monitoring/summary").json()["decisions"].values()) >= 30:
                break
            time.sleep(0.05)
        assert sum(c.get("/api/monitoring/summary").json()["decisions"].values()) == 30
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/api/health").json() == {"status": "ok"}
        assert c.get("/").text == "<div id=root></div>"
        assert c.get("/queue").text == "<div id=root></div>"  # client-side route
        assert c.get("/cases/abc").text == "<div id=root></div>"
        assert "console.log" in c.get("/assets/app.js").text
        assert c.get("/..%2F..%2Fpyproject.toml").text == "<div id=root></div>"  # no traversal
        assert c.get("/api/cases").status_code == 200


@pytest.mark.unit
def test_in_memory_case_store_is_bounded() -> None:
    import asyncio

    store = InMemoryCaseStore(max_cases=10)

    async def fill() -> None:
        for i in range(25):
            await store.open_case(make_case(f"c{i:02d}", float(i), minutes=i))

    asyncio.run(fill())
    kept = sorted(c.case_id for c in asyncio.run(store.list_cases(None, 100)))
    assert len(kept) <= 10
    assert "c24" in kept  # newest kept
    assert "c00" not in kept  # oldest evicted
    assert {a.case_id for a in store.audit} == set(kept)


@pytest.mark.unit
def test_window_defaults_to_the_fixture() -> None:
    w = load_window(Settings())
    assert w.height == 2000
    assert w["event_time"].is_sorted()
    assert isinstance(w, pl.DataFrame)
