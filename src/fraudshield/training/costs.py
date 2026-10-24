"""Cost-sensitive decision policy (``configs/costs.yaml``).

Pure numpy/polars, shared by evaluation and serving. With ``p`` = calibrated P(fraud) and ``A`` =
amount, expected costs are::

    approve = p * (A + chargeback_fee)
    block = (1 - p) * (margin_rate * A + churn_penalty)
    review = analyst_cost + (1 - p) * review_friction

Each transaction gets its cheapest decision. When more than ``review_capacity_per_day`` cheapest-
is-review transactions fall on one UTC day, only those with the largest saving
``min(approve, block) - review`` keep review; the rest get the cheaper of approve/block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fraudshield.schema import DECISIONS, Decision

FloatArray = npt.NDArray[np.float64]
# Fixed width that fits every decision: numpy would otherwise size string arrays to their
# longest value and silently truncate e.g. "approve" -> "approv" on assignment.
DECISION_DTYPE: Final = "<U7"
_APPROVE, _REVIEW, _BLOCK = 0, 1, 2
COST_FIELDS: Final = (
    "chargeback_fee",
    "margin_rate",
    "churn_penalty",
    "analyst_cost",
    "review_friction",
)


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chargeback_fee: float = Field(ge=0)
    margin_rate: float = Field(ge=0)
    churn_penalty: float = Field(ge=0)
    analyst_cost: float = Field(ge=0)
    review_friction: float = Field(ge=0)
    review_capacity_per_day: int | None = Field(default=None, ge=0, description="None = unlimited")

    @classmethod
    def load(cls, path: Path) -> CostConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def scaled(self, field: str, factor: float) -> CostConfig:
        if field not in COST_FIELDS:
            raise ValueError(f"unknown cost field {field}")
        return self.model_copy(update={field: getattr(self, field) * factor})


def expected_costs(p: FloatArray, amount: FloatArray, c: CostConfig) -> FloatArray:
    """(n, 3) expected cost of [approve, review, block]."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    a = np.asarray(amount, dtype=np.float64)
    return np.column_stack(
        [
            p * (a + c.chargeback_fee),
            c.analyst_cost + (1 - p) * c.review_friction,
            (1 - p) * (c.margin_rate * a + c.churn_penalty),
        ]
    )


def decide_one(p: float, amount: float, c: CostConfig) -> Decision:
    """Cheapest decision for a single transaction (capacity is enforced by the caller)."""
    costs = expected_costs(np.array([p]), np.array([amount]), c)[0]
    return DECISIONS[int(np.argmin(costs))]


def limit_reviews(
    decisions: npt.NDArray[np.str_],
    priority: npt.ArrayLike,
    day: npt.ArrayLike,
    capacity: int | None,
    fallback: npt.NDArray[np.str_],
) -> npt.NDArray[np.str_]:
    """Keep at most ``capacity`` reviews per day (highest ``priority`` first; ties by position).

    Reviews over capacity take the matching ``fallback`` decision.
    """
    out = np.asarray(decisions, dtype=DECISION_DTYPE).copy()
    if capacity is None:
        return out
    frame = pl.DataFrame(
        {
            "day": np.asarray(day),
            "review": decisions == "review",
            "prio": np.asarray(priority, dtype=np.float64),
        }
    ).with_row_index("_i")
    ranked = (
        frame.filter(pl.col("review"))
        .sort(["day", "prio", "_i"], descending=[False, True, False])
        .with_columns(r=pl.int_range(1, pl.len() + 1).over("day"))
    )
    over = ranked.filter(pl.col("r") > capacity)["_i"].to_numpy()
    out[over] = fallback[over]
    return out


def decide(
    p: FloatArray, amount: FloatArray, day: npt.ArrayLike, c: CostConfig
) -> npt.NDArray[np.str_]:
    """Vectorised policy with the per-day review capacity."""
    ec = expected_costs(p, amount, c)
    labels = np.asarray(DECISIONS, dtype=DECISION_DTYPE)
    best = labels[np.argmin(ec, axis=1)]
    fallback = np.where(ec[:, _APPROVE] <= ec[:, _BLOCK], "approve", "block").astype(DECISION_DTYPE)
    saving = np.minimum(ec[:, _APPROVE], ec[:, _BLOCK]) - ec[:, _REVIEW]
    return limit_reviews(best, saving, day, c.review_capacity_per_day, fallback)


def realized_costs(
    decision: npt.NDArray[np.str_], y: npt.ArrayLike, amount: FloatArray, c: CostConfig
) -> FloatArray:
    """Cost actually incurred per transaction given the true label."""
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(amount, dtype=np.float64)
    approve = y * (a + c.chargeback_fee)
    block = (1 - y) * (c.margin_rate * a + c.churn_penalty)
    review = c.analyst_cost + (1 - y) * c.review_friction
    out: FloatArray = np.select(
        [decision == "approve", decision == "review", decision == "block"], [approve, review, block]
    )
    return out


def dollar_summary(
    decision: npt.NDArray[np.str_], y: npt.ArrayLike, amount: FloatArray, c: CostConfig
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    a = np.asarray(amount, dtype=np.float64)
    cost = realized_costs(decision, y, a, c)
    fraud = y == 1
    stopped = fraud & (decision != "approve")
    return {
        "total_cost": float(cost.sum()),
        "fraud_amount": float(a[fraud].sum()),
        "fraud_amount_stopped": float(a[stopped].sum()),
        "fraud_amount_missed": float(a[fraud & ~stopped].sum()),
        "legit_blocked": float(((decision == "block") & ~fraud).sum()),
        "n_approve": float((decision == "approve").sum()),
        "n_review": float((decision == "review").sum()),
        "n_block": float((decision == "block").sum()),
    }


def approve_all_cost(y: npt.ArrayLike, amount: FloatArray, c: CostConfig) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float((y * (np.asarray(amount, dtype=np.float64) + c.chargeback_fee)).sum())


def policy_outcome(window: pl.DataFrame, c: CostConfig) -> dict[str, float]:
    """Decisions + dollars for a held-out ``window`` (probability, amount, day, label) under ``c``."""
    p, a, y = (
        window["probability"].to_numpy(),
        window["amount"].to_numpy(),
        window["label"].to_numpy(),
    )
    d = decide(p, a, window["day"].to_numpy(), c)
    out = dollar_summary(d, y, a, c)
    out["approve_all_cost"] = approve_all_cost(y, a, c)
    out["net_savings_vs_approve_all"] = out["approve_all_cost"] - out["total_cost"]
    return out
