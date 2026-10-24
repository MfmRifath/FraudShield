"""Analyst review cases.

``InMemoryCaseStore`` backs tests and the demo container; the Postgres store (tables ``cases``,
``labels``, ``audit_log``) implements the same protocol in Phase 7.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Case(BaseModel):
    case_id: str
    created_at: datetime
    txn: dict[str, Any]
    probability: float | None
    decision: str
    reasons: list[dict[str, Any]]
    expected_loss: float = Field(description="probability x (amount + chargeback fee): queue order")
    model_version: str | None
    status: Literal["open", "labeled"] = "open"
    label: Literal["fraud", "legit"] | None = None
    label_source: Literal["analyst", "chargeback"] | None = None
    analyst_id: str | None = None
    notes: str | None = None
    labeled_at: datetime | None = None


class AuditEntry(BaseModel):
    at: datetime
    case_id: str
    action: str
    actor: str
    detail: dict[str, Any] = Field(default_factory=dict)


class CaseNotFoundError(KeyError):
    pass


class CaseStore(Protocol):
    async def open_case(self, case: Case) -> None: ...

    async def list_cases(self, status: str | None, limit: int) -> list[Case]: ...

    async def get(self, case_id: str) -> Case: ...

    async def label(
        self,
        case_id: str,
        label: Literal["fraud", "legit"],
        analyst_id: str,
        notes: str | None,
        at: datetime,
    ) -> Case: ...

    async def open_count(self) -> int: ...


class InMemoryCaseStore:
    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}
        self.audit: list[AuditEntry] = []

    async def open_case(self, case: Case) -> None:
        if case.case_id in self._cases:
            return  # idempotent: a replayed transaction must not reopen a labelled case
        self._cases[case.case_id] = case
        self.audit.append(
            AuditEntry(at=case.created_at, case_id=case.case_id, action="open", actor="system")
        )

    async def list_cases(self, status: str | None, limit: int) -> list[Case]:
        cases = [c for c in self._cases.values() if status is None or c.status == status]
        cases.sort(key=lambda c: (-c.expected_loss, c.created_at, c.case_id))
        return cases[:limit]

    async def get(self, case_id: str) -> Case:
        try:
            return self._cases[case_id]
        except KeyError as e:
            raise CaseNotFoundError(case_id) from e

    async def label(
        self,
        case_id: str,
        label: Literal["fraud", "legit"],
        analyst_id: str,
        notes: str | None,
        at: datetime,
    ) -> Case:
        case = await self.get(case_id)
        updated = case.model_copy(
            update={"status": "labeled", "label": label, "label_source": "analyst",
                    "analyst_id": analyst_id, "notes": notes, "labeled_at": at}
        )  # fmt: skip
        self._cases[case_id] = updated
        self.audit.append(
            AuditEntry(
                at=at, case_id=case_id, action="label", actor=analyst_id, detail={"label": label}
            )
        )
        return updated

    async def open_count(self) -> int:
        return sum(c.status == "open" for c in self._cases.values())
