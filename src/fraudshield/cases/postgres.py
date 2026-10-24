"""Postgres case store (tables ``cases``, ``labels``, ``audit_log``).

``labels`` is append-only and holds analyst labels (``label_source='analyst'``) and, when they
are ingested, chargebacks (``'chargeback'``). ``build_training_set`` reads analyst labels from it
(``fetch_analyst_labels``). Labelling a case updates ``cases``, appends to ``labels`` and writes
an audit entry in one transaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import orjson
import polars as pl
import psycopg
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from fraudshield.cases.store import AuditEntry, Case, CaseNotFoundError

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id        TEXT PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL,
    uid            TEXT,
    txn            JSONB NOT NULL,
    probability    DOUBLE PRECISION,
    decision       TEXT NOT NULL,
    reasons        JSONB NOT NULL,
    expected_loss  DOUBLE PRECISION NOT NULL,
    model_version  TEXT,
    status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'labeled')),
    label          TEXT CHECK (label IN ('fraud', 'legit')),
    label_source   TEXT CHECK (label_source IN ('analyst', 'chargeback')),
    analyst_id     TEXT,
    notes          TEXT,
    labeled_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS cases_queue ON cases (status, expected_loss DESC, created_at);
CREATE TABLE IF NOT EXISTS labels (
    id            BIGSERIAL PRIMARY KEY,
    txn_id        TEXT NOT NULL,
    label         SMALLINT NOT NULL CHECK (label IN (0, 1)),
    label_source  TEXT NOT NULL CHECK (label_source IN ('analyst', 'chargeback')),
    label_time    TIMESTAMPTZ NOT NULL,
    analyst_id    TEXT,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS labels_txn ON labels (txn_id, label_time);
CREATE TABLE IF NOT EXISTS audit_log (
    id       BIGSERIAL PRIMARY KEY,
    at       TIMESTAMPTZ NOT NULL,
    case_id  TEXT NOT NULL,
    action   TEXT NOT NULL,
    actor    TEXT NOT NULL,
    detail   JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

_COLS = ("case_id", "created_at", "txn", "probability", "decision", "reasons", "expected_loss",
         "model_version", "status", "label", "label_source", "analyst_id", "notes", "labeled_at")  # fmt: skip
_SELECT = sql.SQL("SELECT {} FROM cases").format(sql.SQL(", ").join(map(sql.Identifier, _COLS)))
Pool = AsyncConnectionPool[AsyncConnection[DictRow]]


def _json(v: Any) -> Jsonb:
    return Jsonb(v, dumps=lambda o: orjson.dumps(o, default=str).decode())


def _case(row: DictRow) -> Case:
    return Case.model_validate({k: row[k] for k in _COLS})


class PostgresCaseStore:
    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str, max_size: int = 4) -> PostgresCaseStore:
        pool: Pool = AsyncConnectionPool(
            dsn, min_size=1, max_size=max_size, open=False,
            connection_class=AsyncConnection[DictRow], kwargs={"row_factory": dict_row},
        )  # fmt: skip
        await pool.open(wait=True, timeout=10)
        async with pool.connection() as conn:
            await conn.execute(SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def open_case(self, case: Case) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                "INSERT INTO cases (case_id, created_at, uid, txn, probability, decision, reasons, "
                "expected_loss, model_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (case_id) DO NOTHING",
                (case.case_id, case.created_at, case.txn.get("uid"), _json(case.txn), case.probability,
                 case.decision, _json(case.reasons), case.expected_loss, case.model_version),
            )  # fmt: skip
            if cur.rowcount:  # idempotent: a replayed transaction must not reopen a labelled case
                await conn.execute(
                    "INSERT INTO audit_log (at, case_id, action, actor) VALUES (%s, %s, 'open', 'system')",
                    (case.created_at, case.case_id),
                )

    async def list_cases(self, status: str | None, limit: int) -> list[Case]:
        query = _SELECT + sql.SQL(
            " WHERE (%s::text IS NULL OR status = %s) ORDER BY expected_loss DESC, created_at, case_id LIMIT %s"
        )
        async with self.pool.connection() as conn:
            cur = await conn.execute(query, (status, status, limit))
            return [_case(r) for r in await cur.fetchall()]

    async def get(self, case_id: str) -> Case:
        async with self.pool.connection() as conn:
            cur = await conn.execute(_SELECT + sql.SQL(" WHERE case_id = %s"), (case_id,))
            row = await cur.fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)
        return _case(row)

    async def label(
        self,
        case_id: str,
        label: Literal["fraud", "legit"],
        analyst_id: str,
        notes: str | None,
        at: datetime,
    ) -> Case:
        async with self.pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                "UPDATE cases SET status='labeled', label=%s, label_source='analyst', analyst_id=%s, "
                "notes=%s, labeled_at=%s WHERE case_id=%s",
                (label, analyst_id, notes, at, case_id),
            )
            if not cur.rowcount:
                raise CaseNotFoundError(case_id)
            await conn.execute(
                "INSERT INTO labels (txn_id, label, label_source, label_time, analyst_id, notes) "
                "VALUES (%s, %s, 'analyst', %s, %s, %s)",
                (case_id, 1 if label == "fraud" else 0, at, analyst_id, notes),
            )
            await conn.execute(
                "INSERT INTO audit_log (at, case_id, action, actor, detail) VALUES (%s, %s, 'label', %s, %s)",
                (at, case_id, analyst_id, _json({"label": label, "notes": notes})),
            )
        return await self.get(case_id)

    async def open_count(self) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM cases WHERE status = 'open'")
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def analyst_labels(self) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT txn_id, label, label_time, analyst_id FROM labels "
                "WHERE label_source = 'analyst' ORDER BY label_time"
            )
            return [dict(r) for r in await cur.fetchall()]

    async def audit_trail(self, case_id: str) -> list[AuditEntry]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT at, case_id, action, actor, detail FROM audit_log WHERE case_id = %s ORDER BY id",
                (case_id,),
            )
            return [AuditEntry.model_validate(dict(r)) for r in await cur.fetchall()]


def fetch_analyst_labels(dsn: str) -> pl.DataFrame:
    """Synchronous read for ``build_training_set`` (Prefect worker)."""
    from fraudshield.mlops.training_set import ANALYST_SCHEMA

    with psycopg.Connection.connect(dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT txn_id, label, label_time, analyst_id FROM labels WHERE label_source = 'analyst'"
        ).fetchall()
    return pl.DataFrame([dict(r) for r in rows], schema=ANALYST_SCHEMA)
