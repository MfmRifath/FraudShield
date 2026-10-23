from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from hypothesis import given

from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.rules import Condition, Rule, RuleSet
from tests.strategies import EXTRA_NAMES, canonical_frames

REPO = Path(__file__).resolve().parents[2]
RULES = RuleSet.load(REPO / "configs/rules.yaml")
FS = build_feature_set(EXTRA_NAMES, 5.0)


@pytest.mark.unit
def test_configured_rules_reference_real_features() -> None:
    assert RULES.rules
    assert RULES.features <= set(FS.names)


@pytest.mark.unit
def test_block_beats_review_and_nulls_never_fire() -> None:
    rs = RuleSet(
        rules=(
            Rule(
                name="r",
                action="review",
                reason="r",
                all=(Condition(feature="a", op="ge", value=1),),
            ),
            Rule(
                name="b",
                action="block",
                reason="b",
                all=(Condition(feature="b", op="in", value=("x", "y")),),
            ),
        )
    )
    assert rs.evaluate_row({"a": 2, "b": "x"}).action == "block"
    assert rs.evaluate_row({"a": 2, "b": "z"}).action == "review"
    assert rs.evaluate_row({"a": None, "b": None}).action is None
    assert rs.evaluate_row({"a": float("nan")}).action is None
    r = rs.evaluate_row({"a": 1, "b": "y"})
    assert r.fired == ("r", "b")
    assert r.reasons == ("r", "b")


@pytest.mark.unit
@given(df=canonical_frames)
def test_row_and_frame_evaluation_agree(df: pl.DataFrame) -> None:
    feats = compute_offline(df, FS)
    # make some rules fire on tiny generated frames
    rs = RuleSet(
        rules=(
            *RULES.rules,
            Rule(
                name="burst",
                action="block",
                reason="",
                all=(Condition(feature="uid_cnt_1h", op="ge", value=2),),
            ),
            Rule(
                name="amt",
                action="review",
                reason="",
                all=(Condition(feature="amount", op="gt", value=250),),
            ),
            Rule(
                name="dev",
                action="review",
                reason="",
                all=(Condition(feature="device_type", op="eq", value="mobile"),),
            ),
        )
    )
    frame = rs.evaluate_frame(feats)
    for row, fa, ff in zip(
        feats.to_dicts(), frame["rule_action"], frame["rules_fired"], strict=True
    ):
        r = rs.evaluate_row(row)
        assert (r.action, list(r.fired)) == (fa, list(ff))


@pytest.mark.unit
def test_empty_ruleset() -> None:
    out = RuleSet().evaluate_frame(pl.DataFrame({"x": [1, 2]}))
    assert out["rule_action"].to_list() == [None, None]
