"""Evaluation suite: ranking, calibration and dollar metrics on the test window.

Everything here is computed from (label, probability, decision, amount, day, uid) arrays, so
any model, including the rules-only baseline, is evaluated the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraudshield.training.costs import (
    COST_FIELDS,
    CostConfig,
    approve_all_cost,
    decide,
    dollar_summary,
)

FloatArray = npt.NDArray[np.float64]
FPR_TARGETS: Final = (0.001, 0.005, 0.01)


def recall_at_fpr(y: npt.ArrayLike, p: npt.ArrayLike, fpr_target: float) -> float:
    """Highest recall (TPR) reachable with false-positive rate <= ``fpr_target``."""
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def expected_calibration_error(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> float:
    t, s = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    idx = np.minimum((s * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(s[m].mean() - t[m].mean())
    return float(ece)


def reliability_table(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> pl.DataFrame:
    s = np.asarray(p, dtype=np.float64)
    return (
        pl.DataFrame({"y": np.asarray(y, dtype=np.float64), "p": s})
        .with_columns(bin=pl.min_horizontal((pl.col("p") * bins).floor(), bins - 1).cast(pl.Int64))
        .group_by("bin")
        .agg(n=pl.len(), mean_p=pl.col("p").mean(), frac_pos=pl.col("y").mean())
        .sort("bin")
    )


def card_precision_at_k(
    day: npt.ArrayLike, uid: npt.ArrayLike, p: npt.ArrayLike, y: npt.ArrayLike, k: int = 100
) -> float:
    """Per day: of the k cards with the highest max score, the share with a fraud that day.

    Averaged over test days (Fraud-Detection-Handbook definition of Card Precision@k).
    """
    per_card = (
        pl.DataFrame(
            {
                "day": np.asarray(day),
                "uid": np.asarray(uid),
                "p": np.asarray(p, dtype=np.float64),
                "y": np.asarray(y),
            }
        )
        .group_by(["day", "uid"])
        .agg(p=pl.col("p").max(), fraud=(pl.col("y") == 1).any())
    )
    daily = (
        per_card.sort(["day", "p", "uid"], descending=[False, True, False])
        .group_by("day", maintain_order=True)
        .head(k)
        .group_by("day")
        .agg(precision=pl.col("fraud").cast(pl.Float64).mean())
    )
    return float(daily["precision"].mean()) if daily.height else 0.0  # type: ignore[arg-type]


def ranking_metrics(
    y: npt.ArrayLike, p: npt.ArrayLike, scores: npt.ArrayLike | None = None
) -> dict[str, float]:
    """Ranking metrics on ``scores`` (default ``p``); Brier and ECE on the probabilities ``p``.

    Pass the raw model score as ``scores``: isotonic calibration is a step function, so it
    merges distinct scores into ties (tens of thousands of values into ~100 levels on IEEE-CIS)
    and understates PR-AUC and recall at a fixed false-positive rate. The order it keeps is
    the raw score's order, so ranking quality is measured there.
    """
    t = np.asarray(y)
    s = p if scores is None else scores
    out = {
        "n": float(t.size),
        "n_pos": float(t.sum()),
        "base_rate": float(t.mean()) if t.size else 0.0,
    }
    if 0 < t.sum() < t.size:
        out["pr_auc"] = float(average_precision_score(t, s))
        out["roc_auc"] = float(roc_auc_score(t, s))
        for f in FPR_TARGETS:
            out[f"recall_at_fpr_{f * 100:g}pct"] = recall_at_fpr(t, s, f)
    out["brier"] = float(brier_score_loss(t, np.clip(np.asarray(p, dtype=np.float64), 0, 1)))
    out["ece"] = expected_calibration_error(t, p)
    return out


@dataclass
class ModelEval:
    name: str
    metrics: dict[str, float]
    dollars: dict[str, float]
    segments: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate(
    name: str,
    test: pl.DataFrame,
    p: FloatArray,
    decisions: npt.NDArray[np.str_],
    costs: CostConfig,
    k: int = 100,
    scores: FloatArray | None = None,
) -> ModelEval:
    """``test`` needs columns label, amount, day, uid, is_new_uid.

    ``p`` = calibrated probabilities (calibration metrics; ``decisions`` come from them);
    ``scores`` = the raw model score for ranking metrics (see ``ranking_metrics``).
    """
    y = test["label"].to_numpy()
    amount = test["amount"].to_numpy()
    s = p if scores is None else scores
    metrics = ranking_metrics(y, p, s)
    metrics[f"card_precision_at_{k}"] = card_precision_at_k(
        test["day"].to_numpy(), test["uid"].to_numpy(), s, y, k
    )
    dollars = dollar_summary(decisions, y, amount, costs)
    dollars["approve_all_cost"] = approve_all_cost(y, amount, costs)
    dollars["net_savings_vs_approve_all"] = dollars["approve_all_cost"] - dollars["total_cost"]
    segments = {}
    for seg, mask in (
        ("new_uid", test["is_new_uid"].to_numpy()),
        ("known_uid", ~test["is_new_uid"].to_numpy()),
    ):
        if mask.any():
            segments[seg] = ranking_metrics(y[mask], p[mask], s[mask]) | {
                "total_cost": dollar_summary(decisions[mask], y[mask], amount[mask], costs)[
                    "total_cost"
                ]
            }
    return ModelEval(name, metrics, dollars, segments)


def add_savings_vs(evals: Sequence[ModelEval], baseline: str) -> None:
    base = next(e for e in evals if e.name == baseline).dollars["total_cost"]
    for e in evals:
        e.dollars[f"net_savings_vs_{baseline}"] = base - e.dollars["total_cost"]


def comparison_table(evals: Sequence[ModelEval], k: int = 100) -> pl.DataFrame:
    rows = []
    for e in evals:
        m, d = e.metrics, e.dollars
        rows.append(
            {
                "model": e.name,
                "pr_auc": m.get("pr_auc"),
                "roc_auc": m.get("roc_auc"),
                "recall@0.1%fpr": m.get("recall_at_fpr_0.1pct"),
                "recall@0.5%fpr": m.get("recall_at_fpr_0.5pct"),
                "recall@1%fpr": m.get("recall_at_fpr_1pct"),
                f"card_precision@{k}": m.get(f"card_precision_at_{k}"),
                "brier": m.get("brier"),
                "ece": m.get("ece"),
                "total_cost": d["total_cost"],
                "net_savings_vs_approve_all": d["net_savings_vs_approve_all"],
                "net_savings_vs_rules_only": d.get("net_savings_vs_rules_only"),
                "n_review": d["n_review"],
                "n_block": d["n_block"],
            }
        )
    return pl.DataFrame(rows)


def to_markdown(df: pl.DataFrame) -> str:
    def cell(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([head, sep, *body]) + "\n"


def cost_sensitivity(
    test: pl.DataFrame, p: FloatArray, costs: CostConfig, factors: Sequence[float] = (0.5, 1.0, 2.0)
) -> pl.DataFrame:
    """Net savings vs approve-all when one cost assumption at a time is scaled."""
    y, amount, day = test["label"].to_numpy(), test["amount"].to_numpy(), test["day"].to_numpy()
    rows = []
    for fld in COST_FIELDS:
        for f in factors:
            c = costs.scaled(fld, f)
            d = decide(p, amount, day, c)
            s = dollar_summary(d, y, amount, c)
            rows.append(
                {
                    "parameter": fld,
                    "factor": f,
                    "value": getattr(c, fld),
                    "total_cost": s["total_cost"],
                    "net_savings_vs_approve_all": approve_all_cost(y, amount, c) - s["total_cost"],
                    "n_review": s["n_review"],
                    "n_block": s["n_block"],
                }
            )
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ plots
def plot_reliability(curves: Mapping[str, tuple[npt.ArrayLike, FloatArray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="perfect")
    for name, (y, p) in curves.items():
        t = reliability_table(y, p)
        ax.plot(t["mean_p"], t["frac_pos"], marker="o", lw=1.2, label=name)
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed fraud rate",
        title="Reliability (test)",
    )
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_pr_curves(curves: Mapping[str, tuple[npt.ArrayLike, FloatArray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(np.asarray(y), p)
        ax.plot(
            rec, prec, lw=1.2, label=f"{name} (AP {average_precision_score(np.asarray(y), p):.3f})"
        )
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_cost_sensitivity(sweep: pl.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for fld in sweep["parameter"].unique(maintain_order=True):
        s = sweep.filter(pl.col("parameter") == fld)
        ax.plot(s["factor"], s["net_savings_vs_approve_all"], marker="o", label=fld)
    ax.set(
        xscale="log",
        xlabel="multiplier on assumption",
        ylabel="net savings vs approve-all",
        title="Cost-assumption sensitivity (full model)",
    )
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_shap_summary(
    contributions: FloatArray,
    x: FloatArray,
    feature_names: Sequence[str],
    path: Path,
    max_display: int = 20,
    seed: int = 0,
) -> Path:
    """Beeswarm of exact TreeSHAP values (LightGBM ``pred_contrib``; last column = bias)."""
    import shap

    # all-missing columns (e.g. email fields on Sparkov) carry no signal and break shap's colouring
    keep = ~np.all(np.isnan(x), axis=0)
    shap.summary_plot(
        contributions[:, :-1][:, keep],
        features=x[:, keep],
        feature_names=[n for n, k in zip(feature_names, keep, strict=True) if k],
        max_display=max_display,
        show=False,
        rng=seed,  # explicit: shap's jitter must not depend on the global numpy RNG
    )
    return _save(plt.gcf(), path)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
