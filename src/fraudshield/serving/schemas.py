"""API contracts for the scoring service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudshield.schema import EXTRA_PREFIX, Decision


class ScoreRequest(BaseModel):
    """One transaction to score. ``event_time`` defaults to "now" (UTC) when omitted.

    ``extra`` holds dataset-specific numeric fields by their raw name (e.g. ``{"C1": 2}``); keys
    are case-insensitive and map to the model's ``x_<name>`` features.
    """

    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(min_length=1, max_length=128)
    uid: str = Field(min_length=1, max_length=256)
    amount: float = Field(ge=0, le=1e9)
    event_time: datetime | None = None
    product: str | None = None
    card_network: str | None = None
    card_kind: str | None = None
    p_email_domain: str | None = None
    r_email_domain: str | None = None
    device_type: str | None = None
    device_info: str | None = None
    addr: str | None = None
    extra: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("event_time must be timezone-aware")
        return None if v is None else v.astimezone(UTC)

    def to_row(self, now: datetime) -> dict[str, Any]:
        """Canonical row (same keys as the canonical frame) for feature computation."""
        row = self.model_dump(exclude={"extra", "event_time"})
        row["event_time"] = self.event_time or now
        row["label"] = None
        row["label_available_time"] = None
        for k, v in self.extra.items():
            row[EXTRA_PREFIX + k.lower()] = v
        return row


class ReasonCode(BaseModel):
    code: str
    message: str
    feature: str | None = None
    value: Any = None
    typical: Any = None


Variant = Literal["champion", "challenger", "rules_only"]


class ScoreResponse(BaseModel):
    txn_id: str
    probability: float | None = Field(description="calibrated P(fraud); null when rules-only")
    decision: Decision
    reason_codes: list[ReasonCode] = Field(max_length=3)
    model_version: str | None
    degraded: bool
    latency_ms: float
    variant: Variant
    rules_fired: list[str] = Field(default_factory=list)
    fallback: str | None = Field(default=None, description="why a fallback path was taken")


class LabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["fraud", "legit"]
    analyst_id: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=4000)


class ModelInfo(BaseModel):
    champion: str | None
    challenger: str | None
    canary_pct: int
    dataset: str
