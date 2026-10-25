from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given

from fraudshield.features.definitions import build_feature_set
from fraudshield.features.offline import compute_offline
from fraudshield.reasons import GROUPS, ReasonMapper, describe
from fraudshield.training.costs import CostConfig
from fraudshield.training.evaluate import (
    card_precision_at_k,
    comparison_table,
    cost_sensitivity,
    evaluate,
    expected_calibration_error,
    recall_at_fpr,
    to_markdown,
)
from fraudshield.training.matrix import CategoryEncoder, build_matrix, build_row, typical_values
from tests.strategies import EXTRA_NAMES, canonical_frames

FS = build_feature_set(EXTRA_NAMES, 5.0)


# ------------------------------------------------------------------ matrix
@pytest.mark.unit
def test_encoder_vocab_order_min_count_and_unknowns() -> None:
    df = pl.DataFrame({"c": ["a", "b", "b", "c", "c", "c", None]})
    enc = CategoryEncoder.fit(df, ["c"], min_count=2)
    assert enc.vocab["c"] == ("c", "b")  # most frequent first, "a" below min_count
    codes = build_matrix(pl.DataFrame({"c": ["c", "b", "a", None, "zzz"]}), ["c"], enc)[:, 0]
    assert codes[:2].tolist() == [0.0, 1.0]
    assert all(math.isnan(v) for v in codes[2:])
    assert CategoryEncoder.model_validate_json(enc.model_dump_json()) == enc


@pytest.mark.unit
@given(df=canonical_frames)
def test_build_row_matches_build_matrix(df: pl.DataFrame) -> None:
    feats = compute_offline(df, FS)
    enc = CategoryEncoder.fit(feats, FS.categorical, min_count=1)
    m = build_matrix(feats, FS.names, enc)
    for i, row in enumerate(feats.to_dicts()):
        np.testing.assert_array_equal(build_row(row, FS.names, enc)[0], m[i])


@pytest.mark.unit
def test_typical_values() -> None:
    df = pl.DataFrame({"n": [1.0, 2.0, 10.0, None], "c": ["x", "y", "y", None]})
    assert typical_values(df, ["n", "c"], ["c"]) == {"n": 2.0, "c": "y"}


# ------------------------------------------------------------------ reasons
@pytest.mark.unit
def test_every_feature_has_a_known_reason_group() -> None:
    for f in FS.features:
        group, label = describe(f)
        assert group in GROUPS
        assert label


@pytest.mark.unit
def test_top_reasons_groups_positive_contributions() -> None:
    names = ["uid_cnt_10m", "uid_cnt_1h", "amount", "hour", "te_addr"]
    rm = ReasonMapper.from_feature_set(FS, names, {"uid_cnt_10m": 1.0, "amount": 50.0})
    contrib = np.array([0.4, 0.5, 0.6, -2.0, 0.05, -3.0])  # last = bias
    top = rm.top(contrib, {"uid_cnt_10m": 7.0, "uid_cnt_1h": 9.0, "amount": 800.0})
    assert [r.code for r in top] == ["VELOCITY", "AMOUNT", "RISKY_ATTRIBUTE"]  # 0.9, 0.6, 0.05
    assert top[0].feature == "uid_cnt_1h"  # largest member of the group
    assert top[0].contribution == pytest.approx(0.9)
    assert "typical" in top[1].message
    assert all(r.code != "TIMING" for r in top)  # negative contributions never become reasons
    assert len(rm.top(np.array([-1.0, -1, -1, -1, -1, 0]), {})) == 0


@pytest.mark.unit
def test_reason_mapping_roundtrip() -> None:
    rm = ReasonMapper.from_feature_set(FS, FS.names, {})
    back = ReasonMapper.from_mapping(rm.mapping(), {})
    assert back.groups == rm.groups
    assert back.labels == rm.labels
    with pytest.raises(ValueError, match="unknown reason groups"):
        ReasonMapper(["x"], ["NOPE"], ["x"], {})


# ------------------------------------------------------------------ evaluation
@pytest.mark.unit
def test_recall_at_fpr_and_ece() -> None:
    y = np.array([0] * 1000 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 1000), np.full(10, 0.9)])
    assert recall_at_fpr(y, p, 0.001) == 1.0
    p2 = np.concatenate([np.linspace(0, 0.95, 1000), np.full(10, 0.9)])
    assert recall_at_fpr(y, p2, 0.001) == 0.0  # 53 negatives score above 0.9 -> FPR 5.3%
    assert expected_calibration_error(np.array([0, 1] * 50), np.full(100, 0.5)) == pytest.approx(
        0.0
    )


@pytest.mark.unit
def test_card_precision_at_k_hand_computed() -> None:
    day = [1, 1, 1, 1, 2, 2]
    uid = ["a", "a", "b", "c", "a", "d"]
    p = [0.9, 0.1, 0.8, 0.2, 0.3, 0.7]
    y = [0, 1, 0, 1, 0, 0]
    # day 1, top-2 cards by max score: a (0.9, has fraud), b (0.8, none) -> 0.5 ; day 2: d, a -> 0.0
    assert card_precision_at_k(day, uid, p, y, k=2) == pytest.approx(0.25)


@pytest.mark.unit
def test_evaluate_comparison_and_sweep() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    test = pl.DataFrame(
        {
            "label": (rng.random(n) < 0.05).astype(int),
            "amount": rng.lognormal(4, 1, n),
            "day": rng.integers(1, 10, n),
            "uid": rng.integers(0, 200, n).astype(str),
            "is_new_uid": rng.random(n) < 0.1,
        }
    )
    p = np.clip(test["label"].to_numpy() * 0.5 + rng.random(n) * 0.5, 0, 1)
    costs = CostConfig(chargeback_fee=25, margin_rate=0.03, churn_penalty=10, analyst_cost=4,
                       review_friction=2, review_capacity_per_day=20)  # fmt: skip
    from fraudshield.training.costs import decide

    ev = evaluate(
        "m", test, p, decide(p, test["amount"].to_numpy(), test["day"].to_numpy(), costs), costs
    )
    assert 0 < ev.metrics["pr_auc"] <= 1
    assert set(ev.segments) == {"new_uid", "known_uid"}
    assert ev.dollars["n_review"] <= 20 * 9
    table = comparison_table([ev])
    assert table["model"].to_list() == ["m"]
    md = to_markdown(table)
    assert md.startswith("| model |")
    sweep = cost_sensitivity(test, p, costs)
    assert sweep.height == 5 * 3
    one = sweep.filter((pl.col("parameter") == "analyst_cost") & (pl.col("factor") == 1.0))
    assert one["total_cost"][0] == pytest.approx(ev.dollars["total_cost"])


@pytest.mark.unit
def test_ranking_metrics_use_raw_scores_not_isotonic_ties() -> None:
    """Isotonic calibration merges scores into ties; ranking must be measured on the raw score."""
    from fraudshield.training.calibrate import fit_calibrator
    from fraudshield.training.evaluate import ranking_metrics

    rng = np.random.default_rng(3)
    n = 20_000
    y = (rng.random(n) < 0.05).astype(np.int8)
    raw = y * 1.5 + rng.normal(size=n)  # informative, all distinct
    cal = fit_calibrator(raw[: n // 2], y[: n // 2], platt_below_positives=10)
    p = cal.apply(raw[n // 2 :])
    yt, rt = y[n // 2 :], raw[n // 2 :]
    assert len(np.unique(p)) < len(np.unique(rt)) / 10  # isotonic plateaus
    on_p, on_raw = ranking_metrics(yt, p), ranking_metrics(yt, p, rt)
    assert on_raw["pr_auc"] > on_p["pr_auc"]  # ties understate ranking quality
    assert on_raw["recall_at_fpr_0.1pct"] >= on_p["recall_at_fpr_0.1pct"]
    assert (on_raw["brier"], on_raw["ece"]) == (on_p["brier"], on_p["ece"])  # calibration: on p
