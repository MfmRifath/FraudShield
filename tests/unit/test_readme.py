"""README headline metrics: generated from reports/, never stale, pending until reports exist."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fraudshield.readme import END, PENDING_REAL, START, build_rows, main, render, updated

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_committed_readme_metrics_are_current() -> None:
    assert (
        main(["--readme", str(REPO / "README.md"), "--reports", str(REPO / "reports"), "--check"])
        == 0
    )


@pytest.mark.unit
def test_model_rows_are_pending_without_a_training_report(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)
    assert [r.value for r in rows] == [PENDING_REAL]  # nothing else exists: no invented numbers


@pytest.mark.unit
def test_model_rows_come_from_metrics_json(tmp_path: Path) -> None:
    (tmp_path / "ieee_cis").mkdir()
    full = {"metrics": {"pr_auc": 0.6123, "recall_at_fpr_1pct": 0.4567},
            "dollars": {"net_savings_vs_approve_all": 123456.7}}  # fmt: skip
    base = {"metrics": {"pr_auc": 0.2, "recall_at_fpr_1pct": 0.1}, "dollars": {}}
    (tmp_path / "ieee_cis" / "metrics.json").write_text(
        json.dumps({"evals": {"lgbm_full": full, "rules_only": base}})
    )
    rows = {r.metric: r for r in build_rows(tmp_path)}
    assert rows["PR-AUC, test window (IEEE-CIS)"].value == "0.612 (rules-only 0.200)"
    assert rows["Recall at 1 % false-positive rate"].value == "45.7% (rules-only 10.0%)"
    assert rows["Net savings vs approve-all, test window"].value == "$123,457"
    assert all(r.source == "reports/ieee_cis/metrics.json" for r in rows.values())


@pytest.mark.unit
def test_serving_and_ops_rows_read_the_committed_reports(tmp_path: Path) -> None:
    for rel in ("benchmarks/serving_inprocess.json", "loadtest/results.json",
                "drift/latest.json", "demo/demo-check.json"):  # fmt: skip
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "reports" / rel, tmp_path / rel)
    rows = build_rows(tmp_path)
    areas = [r.area for r in rows]
    assert areas.count("Serving") == 3
    assert {"Resilience", "Monitoring", "Deployment"} <= set(areas)
    lt = json.loads((tmp_path / "loadtest/results.json").read_text())
    normal = next(s for s in lt["scenarios"] if s["name"] == "normal")
    assert f"{normal['client']['rps']:.1f} req/s" in rows[1].value + rows[2].value


@pytest.mark.unit
def test_updated_replaces_only_the_marked_block() -> None:
    text = f"intro\n{START}\nold\n{END}\noutro\n"
    new = updated(text, render([]))
    assert new.startswith("intro\n")
    assert new.endswith("\noutro\n")
    assert "old" not in new
    with pytest.raises(ValueError, match="markers"):
        updated("no markers", "x")
