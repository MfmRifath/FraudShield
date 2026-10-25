"""Drift detection over the served-traffic log (decisions Parquet written by the sink).

Signals, each comparing a *current* window to a *reference* window (event time):

* **feature drift**: Evidently ``DataDriftPreset`` on the champion's top-k features by gain;
  the drifted share comes from Evidently's ``DriftedColumnsCount``;
* **score drift**: PSI of served probabilities (10 reference-quantile bins);
* **decision mix**: approve / review / block shares (reported, not a trigger);
* **performance**: PR-AUC of served probabilities against labels *matured* by ``as_of``
  (skipped with a reason when either window lacks positives).

``should_retrain`` follows the spec: drifted share > 0.3 OR PSI > 0.2 OR PR-AUC drop > 10 %.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import orjson
import polars as pl

from fraudshield.config import MonitoringConfig
from fraudshield.model_bundle import ModelBundle

log = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]


def load_decisions(stream_dir: Path) -> pl.DataFrame:
    files = sorted((stream_dir / "decisions").rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files).unique("txn_id", keep="last").sort("event_time")


def load_labels(stream_dir: Path) -> pl.DataFrame:
    files = sorted((stream_dir / "labels").rglob("*.parquet"))
    if not files:
        return pl.DataFrame(
            schema={
                "txn_id": pl.String,
                "label": pl.Int8,
                "label_available_time": pl.Datetime("us", "UTC"),
            }
        )
    return pl.read_parquet(files).unique("txn_id", keep="last")


def expand_features(decisions: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Decode the JSON feature column into ``columns`` (missing keys -> null)."""
    rows = [orjson.loads(f) for f in decisions["features"].to_list()]
    return pl.DataFrame([{c: r.get(c) for c in columns} for r in rows], infer_schema_length=None)


def top_features(bundle: ModelBundle, k: int) -> list[str]:
    booster = bundle.boosters["full"]
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(-np.asarray(gains), kind="stable")
    names = booster.feature_name()
    return [names[i] for i in order if gains[i] > 0][:k]


def psi(
    reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = 10, eps: float = 1e-4
) -> float:
    """Population stability index with reference-quantile bins."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / ref.size
    c = np.histogram(cur, edges)[0] / cur.size
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def decision_mix(decisions: pl.DataFrame) -> dict[str, float]:
    n = max(decisions.height, 1)
    counts = dict(decisions.group_by("decision").len().iter_rows())
    return {d: counts.get(d, 0) / n for d in ("approve", "review", "block")}


def matured_pr_auc(
    window: pl.DataFrame, labels: pl.DataFrame, as_of: datetime, min_pos: int
) -> tuple[float | None, int, int]:
    from sklearn.metrics import average_precision_score

    joined = window.filter(pl.col("probability").is_not_null()).join(
        labels.filter(pl.col("label_available_time") <= as_of).select("txn_id", "label"),
        on="txn_id",
        how="inner",
    )
    n, pos = joined.height, int(joined["label"].sum()) if joined.height else 0
    if pos < min_pos or pos == n:
        return None, n, pos
    return (
        float(
            average_precision_score(joined["label"].to_numpy(), joined["probability"].to_numpy())
        ),
        n,
        pos,
    )


def evidently_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: list[str],
    categorical: set[str],
    drift_share: float,
) -> tuple[float, list[dict[str, Any]], Any]:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # constant/empty columns break statistical tests; they carry no drift signal
    usable = [
        c
        for c in columns
        if reference[c].drop_nulls().n_unique() > 0 and current[c].drop_nulls().n_unique() > 0
    ]
    num = [c for c in usable if c not in categorical]
    cat = [c for c in usable if c in categorical]
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    ref_ds = Dataset.from_pandas(reference.select(usable).to_pandas(), data_definition=dd)
    cur_ds = Dataset.from_pandas(current.select(usable).to_pandas(), data_definition=dd)
    snapshot = Report([DataDriftPreset(columns=usable, drift_share=drift_share)]).run(
        current_data=cur_ds, reference_data=ref_ds
    )
    share, per_column = 0.0, []
    for m in snapshot.dict()["metrics"]:
        cfg, value = m.get("config", {}), m.get("value")
        if cfg.get("type", "").endswith("DriftedColumnsCount"):
            share = float(value["share"])
        elif cfg.get("type", "").endswith("ValueDrift"):
            method, thr = str(cfg.get("method")), float(cfg.get("threshold"))
            drifted = value < thr if "p_value" in method else value >= thr
            per_column.append(
                {
                    "column": cfg["column"],
                    "method": method,
                    "score": float(value),
                    "threshold": thr,
                    "drifted": bool(drifted),
                }
            )
    # columns dropped as unusable cannot drift; rescale the share to all requested columns
    share = share * len(usable) / len(columns) if columns else 0.0
    return share, per_column, snapshot


@dataclass
class DriftResult:
    as_of: str
    reference_window: tuple[str, str]
    current_window: tuple[str, str]
    rows: dict[str, int]
    features: list[str]
    drift_share: float | None
    drifted_columns: list[dict[str, Any]] = field(default_factory=list)
    score_psi: float | None = None
    decision_mix: dict[str, dict[str, float]] = field(default_factory=dict)
    pr_auc: dict[str, Any] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    report_html: str | None = None

    @property
    def should_retrain(self) -> bool:
        return bool(self.triggers)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"should_retrain": self.should_retrain}


def detect_drift(
    decisions: pl.DataFrame,
    labels: pl.DataFrame,
    bundle: ModelBundle,
    cfg: MonitoringConfig,
    as_of: datetime | None = None,
    out_dir: Path | None = None,
    reference_days: float | None = None,
    current_days: float | None = None,
    performance_days: float | None = None,
    performance_lag_days: float | None = None,
) -> DriftResult:
    if decisions.is_empty():
        raise ValueError("no served decisions logged yet")
    end: datetime = as_of or decisions["event_time"].max()  # type: ignore[assignment]
    cur_start = end - timedelta(days=current_days or cfg.current_days)
    ref_start = cur_start - timedelta(days=reference_days or cfg.reference_days)
    t = pl.col("event_time")
    cur = decisions.filter((t > cur_start) & (t <= end))
    ref = decisions.filter((t > ref_start) & (t <= cur_start))
    res = DriftResult(
        as_of=end.isoformat(),
        reference_window=(ref_start.isoformat(), cur_start.isoformat()),
        current_window=(cur_start.isoformat(), end.isoformat()),
        rows={"reference": ref.height, "current": cur.height},
        features=top_features(bundle, cfg.top_k_features),
        drift_share=None,
    )
    if min(ref.height, cur.height) < cfg.min_rows:
        res.skipped.append(
            f"fewer than {cfg.min_rows} rows in a window ({ref.height} / {cur.height})"
        )
        return res

    categorical = set(bundle.spec.encoder.vocab)
    share, per_col, snapshot = evidently_drift(
        expand_features(ref, res.features), expand_features(cur, res.features), res.features, categorical,
        cfg.drift_share_threshold,
    )  # fmt: skip
    res.drift_share, res.drifted_columns = share, per_col
    res.score_psi = psi(
        ref["probability"].drop_nulls().to_numpy(), cur["probability"].drop_nulls().to_numpy()
    )
    res.decision_mix = {"reference": decision_mix(ref), "current": decision_mix(cur)}
    # performance: the latest window whose labels have matured vs the window before it
    lag = cfg.performance_lag_days if performance_lag_days is None else performance_lag_days
    pdays = cfg.performance_days if performance_days is None else performance_days
    p_end = end - timedelta(days=lag)
    p_cur_start, p_ref_start = p_end - timedelta(days=pdays), p_end - timedelta(days=2 * pdays)
    p_cur = decisions.filter((t > p_cur_start) & (t <= p_end))
    p_ref = decisions.filter((t > p_ref_start) & (t <= p_cur_start))
    pr_ref = matured_pr_auc(p_ref, labels, end, cfg.min_label_positives)
    pr_cur = matured_pr_auc(p_cur, labels, end, cfg.min_label_positives)
    res.pr_auc = {
        "reference": pr_ref[0],
        "current": pr_cur[0],
        "labelled": {"reference": pr_ref[1:], "current": pr_cur[1:]},
        "windows": {
            "reference": [p_ref_start.isoformat(), p_cur_start.isoformat()],
            "current": [p_cur_start.isoformat(), p_end.isoformat()],
        },
    }

    if share > cfg.drift_share_threshold:
        res.triggers.append(
            f"feature drift: {share:.0%} of top-{len(res.features)} features drifted (> {cfg.drift_share_threshold:.0%})"
        )
    if res.score_psi > cfg.psi_threshold:  # NaN (empty window) compares False
        res.triggers.append(f"score drift: PSI {res.score_psi:.3f} > {cfg.psi_threshold}")
    if pr_ref[0] is None or pr_cur[0] is None:
        res.skipped.append("PR-AUC check: too few matured labelled positives in a window")
    else:
        drop = (pr_ref[0] - pr_cur[0]) / pr_ref[0] if pr_ref[0] > 0 else 0.0
        res.pr_auc["relative_drop"] = drop
        if drop > cfg.pr_auc_drop_threshold:
            res.triggers.append(
                f"performance: matured-label PR-AUC dropped {drop:.0%} (> {cfg.pr_auc_drop_threshold:.0%})"
            )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = end.strftime("%Y%m%dT%H%M%SZ")
        html = out_dir / f"drift-{stamp}.html"
        snapshot.save_html(str(html))
        res.report_html = str(html)
        (out_dir / f"drift-{stamp}.json").write_text(
            json.dumps(res.as_dict(), indent=2, default=str) + "\n"
        )
        (out_dir / "latest.json").write_text(
            json.dumps(res.as_dict(), indent=2, default=str) + "\n"
        )
    return res
