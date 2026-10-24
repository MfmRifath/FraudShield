"""Promotion gate: the challenger must beat or match the champion on every check.

1. ``pr_auc``: challenger PR-AUC >= champion PR-AUC - tolerance (test window).
2. ``net_savings``: challenger net savings vs approve-all >= champion's (same costs/capacity).
3. ``latency``: single-row p95 of ``ModelBundle.score_one`` <= max ms.
4. ``parity``: the challenger was trained on the feature set serving computes, and the online
   path reproduces offline features (max |diff| < tolerance) on a canonical sample.

Without a champion (bootstrap), checks 1-2 compare against nothing and pass. The shadow
comparison (decision agreement, |Δp|) is reported but not gating.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from fraudshield.config import MLOpsConfig
from fraudshield.features.definitions import FeatureSet
from fraudshield.features.parity import parity_report
from fraudshield.model_bundle import ModelBundle
from fraudshield.training.costs import CostConfig, decide
from fraudshield.training.evaluate import evaluate


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    detail: str


@dataclass
class GateResult:
    checks: list[Check]
    comparison: dict[str, dict[str, float]]
    shadow: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "comparison": self.comparison,
            "shadow": self.shadow,
        }

    def to_markdown(self) -> str:
        rows = "\n".join(
            f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {_f(c.value)} | {_f(c.threshold)} | {c.detail} |"
            for c in self.checks
        )
        cmp = "\n".join(
            f"| {m} | {_f(v.get('pr_auc'))} | {_f(v.get('net_savings'))} | {_f(v.get('n_review'))} | {_f(v.get('n_block'))} |"
            for m, v in self.comparison.items()
        )
        sh = ", ".join(f"{k} = {_f(v)}" for k, v in self.shadow.items()) or "n/a"
        return f"""**Gate: {"PASSED" if self.passed else "FAILED"}**

| check | result | value | threshold | detail |
|---|---|---|---|---|
{rows}

| model | PR-AUC | net savings | reviews | blocks |
|---|---|---|---|---|
{cmp}

Shadow comparison (test window, not gating): {sh}
"""


def _f(v: float | None) -> str:
    if v is None:
        return "n/a"
    if float(v).is_integer() and abs(v) >= 1:
        return f"{v:,.0f}"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.1e}"
    return f"{v:.4f}" if abs(v) < 100 else f"{v:,.0f}"


def _window_metrics(
    bundle: ModelBundle, test: pl.DataFrame, costs: CostConfig
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    p = bundle.predict_proba(test)
    d = decide(p, test["amount"].to_numpy(), test["day"].to_numpy(), costs)
    ev = evaluate("m", test, p, d, costs)
    m = {"pr_auc": ev.metrics.get("pr_auc", float("nan")), "net_savings": ev.dollars["net_savings_vs_approve_all"],
         "n_review": ev.dollars["n_review"], "n_block": ev.dollars["n_block"]}  # fmt: skip
    return m, p, d


def single_row_p95_ms(bundle: ModelBundle, rows: list[dict[str, Any]]) -> float:
    for r in rows[:20]:  # warm-up
        bundle.score_one(r)
    times = []
    for r in rows:
        t0 = time.perf_counter()
        bundle.score_one(r)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(times, 95))


def run_gate(
    challenger: ModelBundle,
    champion: ModelBundle | None,
    test: pl.DataFrame,
    parity_canonical: pl.DataFrame,
    serving_fs: FeatureSet,
    costs: CostConfig,
    cfg: MLOpsConfig,
) -> GateResult:
    """``test``: labelled test-window feature rows with ``day``, ``uid``, ``amount``, ``is_new_uid``."""
    ch_m, ch_p, ch_d = _window_metrics(challenger, test, costs)
    comparison = {"challenger": ch_m}
    shadow: dict[str, float] = {}
    if champion is not None:
        cp_m, cp_p, cp_d = _window_metrics(champion, test, costs)
        comparison["champion"] = cp_m
        shadow = {
            "decision_agreement": float((ch_d == cp_d).mean()),
            "mean_abs_dp": float(np.abs(ch_p - cp_p).mean()),
        }
        pr = Check("pr_auc", ch_m["pr_auc"] >= cp_m["pr_auc"] - cfg.pr_auc_tolerance, ch_m["pr_auc"],
                   cp_m["pr_auc"] - cfg.pr_auc_tolerance, f"champion {cp_m['pr_auc']:.4f} - {cfg.pr_auc_tolerance}")  # fmt: skip
        sav = Check("net_savings", ch_m["net_savings"] >= cp_m["net_savings"], ch_m["net_savings"], cp_m["net_savings"],
                    "challenger >= champion (test window)")  # fmt: skip
    else:
        pr = Check("pr_auc", True, ch_m["pr_auc"], None, "no champion (bootstrap)")
        sav = Check("net_savings", True, ch_m["net_savings"], None, "no champion (bootstrap)")

    rows = test.head(cfg.latency_bench_rows).to_dicts()
    p95 = single_row_p95_ms(challenger, rows)
    lat = Check(
        "latency",
        p95 <= cfg.max_single_row_p95_ms,
        p95,
        cfg.max_single_row_p95_ms,
        f"single-row p95 over {len(rows)} rows (ms)",
    )

    trained_on = challenger.spec.metadata.get("feature_set_hash")
    if trained_on != serving_fs.spec_hash():
        par = Check(
            "parity",
            False,
            None,
            cfg.parity_tolerance,
            f"feature set {str(trained_on)[:12]} != serving {serving_fs.spec_hash()[:12]}",
        )
    else:
        worst = max(parity_report(parity_canonical.head(cfg.parity_rows), serving_fs).values())
        par = Check(
            "parity",
            worst < cfg.parity_tolerance,
            worst,
            cfg.parity_tolerance,
            f"online vs offline on {min(cfg.parity_rows, parity_canonical.height)} events",
        )
    return GateResult([pr, sav, lat, par], comparison, shadow)
