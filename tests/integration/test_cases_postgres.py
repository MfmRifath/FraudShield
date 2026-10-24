"""Postgres case store against the compose ``postgres`` service (mlops profile).

Each test works in its own schema, so reruns and parallel stacks never see each other's rows.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from fraudshield.cases.postgres import PostgresCaseStore
from fraudshield.flows.flows import load_analyst_labels
from fraudshield.serving.app import create_app
from fraudshield.synthetic import synthetic_canonical
from tests.case_contract import assert_reaches_training_set, check_case_store
from tests.serving_support import (
    ALWAYS_REVIEW_RULES,
    make_bundle,
    payload,
    serving_settings,
    write_yaml,
)

pytestmark = [pytest.mark.integration, pytest.mark.mlops]
BASE_DSN = os.environ.get(
    "FS_IT_PG_DSN", "postgresql://fraudshield:fraudshield@postgres:5432/fraudshield"
)


@pytest.fixture
def dsn() -> Iterator[str]:
    schema = f"it_{uuid.uuid4().hex[:10]}"
    with psycopg.Connection.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")  # generated name, not user input
    yield f"{BASE_DSN}?options=-csearch_path%3D{schema}"
    with psycopg.Connection.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_postgres_case_store_contract(dsn: str) -> None:
    async def run() -> None:
        store = await PostgresCaseStore.connect(dsn)
        try:
            await check_case_store(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_label_in_ui_appears_in_next_training_set_postgres(dsn: str, tmp_path: Path) -> None:
    """Phase 7 DoD on the real store: API label -> ``labels`` table -> build_training_set."""
    canon = synthetic_canonical(10_000, seed=0).head(200)
    base = serving_settings(
        tmp_path, rules=write_yaml(tmp_path / "rules.yaml", ALWAYS_REVIEW_RULES)
    )
    bundle = make_bundle(base, tmp_path / "bundle")
    s = serving_settings(tmp_path, rules=tmp_path / "rules.yaml", bundle_dir=bundle).model_copy(
        update={"cases": base.cases.model_copy(update={"store": "postgres", "dsn": SecretStr(dsn)})}
    )
    with TestClient(create_app(s)) as c:
        for row in canon.to_dicts():
            c.post("/score", json=payload(row))
        target = c.get("/cases", params={"limit": 1}).json()[0]["case_id"]
        r = c.post(f"/cases/{target}/label", json={"label": "fraud", "analyst_id": "ana"})
        assert r.status_code == 200
        assert [a["action"] for a in c.get(f"/cases/{target}/audit").json()] == ["open", "label"]
    analyst = load_analyst_labels(s)  # what the build-training-set flow reads
    assert analyst["txn_id"].to_list() == [target]
    assert analyst.schema["label_time"] == pl.Datetime("us", "UTC")
    assert_reaches_training_set(canon, analyst, target)
