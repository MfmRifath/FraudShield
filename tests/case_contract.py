"""Behaviour every ``CaseStore`` must have (run against InMemory and Postgres)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from fraudshield.cases.store import Case, CaseNotFoundError, CaseStore
from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.mlops.training_set import build_training_set
from fraudshield.synthetic import EXTRAS

T0 = datetime(2021, 3, 1, 12, tzinfo=UTC)


def make_case(case_id: str, expected_loss: float, minutes: int = 0) -> Case:
    return Case(
        case_id=case_id,
        created_at=T0 + timedelta(minutes=minutes),
        txn={
            "txn_id": case_id,
            "uid": f"u-{case_id}",
            "amount": 12.5,
            "event_time": T0.isoformat(),
        },
        probability=0.4,
        decision="review",
        reasons=[{"group": "amount", "contribution": 0.3}],
        expected_loss=expected_loss,
        model_version="m1",
    )


async def check_case_store(store: CaseStore) -> None:
    await store.open_case(make_case("a", 10.0))
    await store.open_case(make_case("b", 30.0, minutes=1))
    await store.open_case(make_case("c", 30.0, minutes=2))
    await store.open_case(make_case("a", 99.0))  # replayed: ignored
    queue = await store.list_cases("open", 10)
    assert [c.case_id for c in queue] == ["b", "c", "a"]  # expected loss desc, then oldest first
    assert queue[0].txn["uid"] == "u-b"
    assert queue[0].reasons == [{"group": "amount", "contribution": 0.3}]
    assert await store.open_count() == 3

    at = T0 + timedelta(hours=1)
    labelled = await store.label("b", "fraud", "ana", "card testing", at)
    assert (labelled.status, labelled.label, labelled.label_source) == (
        "labeled",
        "fraud",
        "analyst",
    )
    assert (labelled.analyst_id, labelled.notes, labelled.labeled_at) == ("ana", "card testing", at)
    await store.open_case(make_case("b", 1.0))  # a replay must not reopen a labelled case
    assert (await store.get("b")).status == "labeled"
    assert [c.case_id for c in await store.list_cases("open", 10)] == ["c", "a"]
    assert [c.case_id for c in await store.list_cases(None, 2)] == ["b", "c"]
    assert await store.open_count() == 2

    await store.label("b", "legit", "bob", None, at + timedelta(minutes=5))  # relabel
    labels = await store.analyst_labels()
    assert [(x["txn_id"], x["label"], x["analyst_id"]) for x in labels][-1] == ("b", 0, "bob")
    assert {x["txn_id"] for x in labels} == {"b"}

    trail = await store.audit_trail("b")
    assert [(a.action, a.actor) for a in trail] == [
        ("open", "system"),
        ("label", "ana"),
        ("label", "bob"),
    ]
    assert trail[1].detail == {"label": "fraud", "notes": "card testing"}
    assert await store.audit_trail("zzz") == []
    with pytest.raises(CaseNotFoundError):
        await store.get("zzz")
    with pytest.raises(CaseNotFoundError):
        await store.label("zzz", "fraud", "ana", None, at)


def assert_reaches_training_set(canon: pl.DataFrame, analyst: pl.DataFrame, txn_id: str) -> None:
    """The analyst label for ``txn_id`` is in the next training set (its chargeback has not matured)."""
    immature = pl.col("txn_id") == txn_id
    feats = compute_offline(canon, build_feature_set(EXTRAS, 50.0)).with_columns(
        label=pl.when(immature).then(None).otherwise(pl.col("label")),
        label_available_time=pl.when(immature).then(None).otherwise(pl.col("label_available_time")),
    )
    out, stats = build_training_set(feats, analyst, as_of=datetime.now(UTC) + timedelta(seconds=1))
    row = out.filter(pl.col("txn_id") == txn_id).row(0, named=True)
    assert (row["label"], row["label_source"]) == (1, "analyst")
    assert stats.analyst_labels == 1
