from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from fraudshield.training.costs import (
    CostConfig,
    approve_all_cost,
    decide,
    decide_one,
    dollar_summary,
    expected_costs,
    limit_reviews,
    realized_costs,
)

REPO = Path(__file__).resolve().parents[2]
C = CostConfig(
    chargeback_fee=25, margin_rate=0.03, churn_penalty=10, analyst_cost=4, review_friction=2,
    review_capacity_per_day=None,
)  # fmt: skip


@pytest.mark.unit
def test_costs_yaml_loads() -> None:
    c = CostConfig.load(REPO / "configs/costs.yaml")
    assert c.review_capacity_per_day == 200


@pytest.mark.unit
def test_expected_cost_formulas() -> None:
    ec = expected_costs(np.array([0.1]), np.array([100.0]), C)[0]
    assert ec[0] == pytest.approx(0.1 * 125)  # approve
    assert ec[1] == pytest.approx(4 + 0.9 * 2)  # review
    assert ec[2] == pytest.approx(0.9 * (3 + 10))  # block


@pytest.mark.unit
def test_extremes() -> None:
    assert decide_one(0.0, 50.0, C) == "approve"
    assert decide_one(1.0, 50.0, C) == "block"
    assert decide_one(0.3, 200.0, C) == "review"  # 67.5 approve, 5.4 review, 11.2 block


probs = hnp.arrays(np.float64, st.integers(1, 60), elements=st.floats(0, 1))


@pytest.mark.unit
@given(p=probs, data=st.data())
def test_decide_without_capacity_is_argmin(p: np.ndarray, data: st.DataObject) -> None:
    amount = data.draw(hnp.arrays(np.float64, p.shape, elements=st.floats(0, 5000)))
    d = decide(p, amount, np.zeros(p.size), C)
    assert list(d) == [decide_one(pi, ai, C) for pi, ai in zip(p, amount, strict=True)]


@pytest.mark.unit
@given(p=probs, cap=st.integers(0, 5), data=st.data())
def test_review_capacity_per_day(p: np.ndarray, cap: int, data: st.DataObject) -> None:
    amount = data.draw(hnp.arrays(np.float64, p.shape, elements=st.floats(0, 5000)))
    day = data.draw(hnp.arrays(np.int64, p.shape, elements=st.integers(0, 2)))
    c = C.model_copy(update={"review_capacity_per_day": cap})
    unlimited = decide(p, amount, day, C)
    capped = decide(p, amount, day, c)
    for dd in np.unique(day):
        m = day == dd
        assert (capped[m] == "review").sum() == min(cap, (unlimited[m] == "review").sum())
    # only would-be reviews change, and they change to the cheaper of approve/block
    changed = capped != unlimited
    assert (unlimited[changed] == "review").all()
    ec = expected_costs(p, amount, C)
    for i in np.flatnonzero(changed):
        assert capped[i] == ("approve" if ec[i, 0] <= ec[i, 2] else "block")


@pytest.mark.unit
def test_capacity_keeps_highest_saving_reviews() -> None:
    p = np.array([0.3, 0.3, 0.3])
    amount = np.array([50.0, 500.0, 200.0])  # saving min(approve, block) - review grows with amount
    c = C.model_copy(update={"review_capacity_per_day": 1})
    # over-capacity reviews fall back to the cheaper of approve/block: here block
    # (A=50: approve 22.5 vs block 8.05; A=200: approve 67.5 vs block 11.2)
    assert list(decide(p, amount, np.zeros(3), c)) == ["block", "review", "block"]


@pytest.mark.unit
def test_limit_reviews_never_truncates_decisions() -> None:
    # regression: an all-"review" input array has dtype <U6; "approve" must not become "approv"
    out = limit_reviews(
        np.array(["review", "review"]), [1, 2], [0, 0], 0, np.array(["approve"] * 2)
    )
    assert list(out) == ["approve", "approve"]


@pytest.mark.unit
def test_limit_reviews_priority_and_ties() -> None:
    d = np.array(["review"] * 4)
    out = limit_reviews(d, [1.0, 5.0, 5.0, 2.0], [0, 0, 0, 0], 2, np.array(["approve"] * 4))
    assert list(out) == ["approve", "review", "review", "approve"]
    assert list(limit_reviews(d, [1, 1, 1, 1], [0] * 4, None, d)) == list(d)


@pytest.mark.unit
def test_realized_costs_and_summary() -> None:
    d = np.array(["approve", "approve", "block", "block", "review", "review"])
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.full(6, 100.0)
    cost = realized_costs(d, y, a, C)
    assert list(cost) == [0.0, 125.0, 13.0, 0.0, 6.0, 4.0]
    s = dollar_summary(d, y, a, C)
    assert s["total_cost"] == pytest.approx(148.0)
    assert s["fraud_amount_stopped"] == 200.0
    assert s["fraud_amount_missed"] == 100.0
    assert s["legit_blocked"] == 1.0
    assert approve_all_cost(y, a, C) == 375.0


@pytest.mark.unit
def test_scaled_rejects_unknown_field() -> None:
    assert C.scaled("analyst_cost", 2).analyst_cost == 8
    with pytest.raises(ValueError, match="unknown"):
        C.scaled("review_capacity_per_day", 2)
