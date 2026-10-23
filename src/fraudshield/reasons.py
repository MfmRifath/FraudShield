"""Human-readable reason codes from per-feature contributions (``pred_contrib=True``).

Features are grouped into reason families. For one prediction, the positive contributions
(pushing towards fraud) are summed per group, the top 3 groups are kept, and each is described
by its single largest feature: "<label>: <value> (typical <typical>)".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from fraudshield.features.definitions import (
    DistinctCount,
    Feature,
    FeatureSet,
    FirstSeen,
    Passthrough,
    Ratio,
    TargetEncoding,
    TimeSinceLast,
    WindowAgg,
)


@dataclass(frozen=True, slots=True)
class ReasonGroup:
    code: str
    title: str


GROUPS: Final[dict[str, ReasonGroup]] = {
    g.code: g
    for g in (
        ReasonGroup("AMOUNT", "Unusual amount"),
        ReasonGroup("VELOCITY", "High transaction velocity"),
        ReasonGroup("SPEND_PATTERN", "Unusual spending pattern for this card"),
        ReasonGroup("RECENCY", "Unusual time since previous transaction"),
        ReasonGroup("IDENTITY_CHURN", "Many devices, emails or addresses for this card"),
        ReasonGroup("SHARED_IDENTITY", "Device or email shared by many cards"),
        ReasonGroup("NEW_IDENTITY", "First use of this device, email or address"),
        ReasonGroup("RISKY_ATTRIBUTE", "Attribute with a high historical fraud rate"),
        ReasonGroup("TIMING", "Unusual time of day or week"),
        ReasonGroup("PROFILE", "Transaction profile"),
    )
}

_ENTITY = {"uid": "this card", "device_info": "this device", "p_email_domain": "this email domain"}
_VALUE = {
    "device_info": "devices",
    "p_email_domain": "email domains",
    "addr": "addresses",
    "uid": "cards",
}
_SINGULAR = {"device_info": "device", "p_email_domain": "email domain", "addr": "address"}


def _window(w: timedelta) -> str:
    s = int(w.total_seconds())
    if s % 86_400 == 0:
        return f"{s // 86_400}d" if s >= 86_400 * 2 else "24h"
    if s % 3_600 == 0:
        return f"{s // 3_600}h"
    return f"{s // 60}m"


def describe(f: Feature) -> tuple[str, str]:
    """(group code, human label) for a feature."""
    if isinstance(f, WindowAgg):
        ent = _ENTITY.get(f.entity, f.entity)
        if f.agg == "count":
            return "VELOCITY", f"transactions on {ent} in the last {_window(f.window)}"
        return "SPEND_PATTERN", f"{f.agg} amount on {ent} in the last {_window(f.window)}"
    if isinstance(f, Ratio):
        return "SPEND_PATTERN", "amount relative to this card's 7-day average"
    if isinstance(f, TimeSinceLast):
        return "RECENCY", "seconds since this card's previous transaction"
    if isinstance(f, DistinctCount):
        group = "IDENTITY_CHURN" if f.entity == "uid" else "SHARED_IDENTITY"
        ent = _ENTITY.get(f.entity, f.entity)
        return group, f"distinct {_VALUE.get(f.value, f.value)} for {ent} in {_window(f.window)}"
    if isinstance(f, FirstSeen):
        return "NEW_IDENTITY", f"first use of this {_SINGULAR.get(f.value, f.value)} by this card"
    if isinstance(f, TargetEncoding):
        return "RISKY_ATTRIBUTE", f"historical fraud rate of this {f.column.replace('_', ' ')}"
    if f.name in ("amount", "log_amount", "amount_cents"):
        return "AMOUNT", {
            "amount": "amount",
            "log_amount": "amount (log)",
            "amount_cents": "cents part of amount",
        }[f.name]
    if f.name in ("hour", "weekday"):
        return "TIMING", {"hour": "hour of day (UTC)", "weekday": "day of week"}[f.name]
    label = f.name.removeprefix("x_") if isinstance(f, Passthrough) else f.name
    return "PROFILE", label.replace("_", " ")


@dataclass(frozen=True, slots=True)
class Reason:
    code: str
    title: str
    feature: str
    label: str
    value: Any
    typical: Any
    contribution: float

    @property
    def message(self) -> str:
        return f"{self.title}: {self.label} = {_fmt(self.value)} (typical {_fmt(self.typical)})"


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.0f}" if v.is_integer() else f"{v:.4g}"
    return str(v)


class ReasonMapper:
    """Maps contributions for ``feature_names`` (model input order) to the top reasons."""

    def __init__(
        self,
        feature_names: Sequence[str],
        groups: Sequence[str],
        labels: Sequence[str],
        typical: Mapping[str, Any],
    ) -> None:
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown reason groups: {sorted(unknown)}")
        self.feature_names = list(feature_names)
        self.groups = list(groups)
        self.labels = list(labels)
        self.typical = dict(typical)
        self._group_index = {g: i for i, g in enumerate(dict.fromkeys(self.groups))}
        self._gidx = np.array([self._group_index[g] for g in self.groups], dtype=np.int64)

    @classmethod
    def from_feature_set(
        cls, fs: FeatureSet, feature_names: Sequence[str], typical: Mapping[str, Any]
    ) -> ReasonMapper:
        by_name = {f.name: f for f in fs.features}
        described = [describe(by_name[n]) for n in feature_names]
        return cls(feature_names, [g for g, _ in described], [lbl for _, lbl in described], typical)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Mapping[str, str]], typical: Mapping[str, Any]
    ) -> ReasonMapper:
        names = list(mapping)
        return cls(
            names,
            [mapping[n]["group"] for n in names],
            [mapping[n]["label"] for n in names],
            typical,
        )

    def mapping(self) -> dict[str, dict[str, str]]:
        """JSON-able feature -> {group, label} map (stored in the model bundle)."""
        return {
            n: {"group": g, "label": lbl}
            for n, g, lbl in zip(self.feature_names, self.groups, self.labels, strict=True)
        }

    def top(
        self,
        contributions: npt.NDArray[np.float64],
        raw_values: Mapping[str, Any],
        n: int = 3,
    ) -> list[Reason]:
        """``contributions``: one row of ``pred_contrib`` output (k features + bias)."""
        c = np.asarray(contributions, dtype=np.float64)[: len(self.feature_names)]
        pos = np.where(c > 0, c, 0.0)
        group_sum = np.bincount(self._gidx, weights=pos, minlength=len(self._group_index))
        order = [int(g) for g in np.argsort(-group_sum, kind="stable") if group_sum[g] > 0][:n]
        codes = list(self._group_index)
        out = []
        for g in order:
            members = np.flatnonzero(self._gidx == g)
            best = int(members[np.argmax(pos[members])])
            name = self.feature_names[best]
            grp = GROUPS[codes[g]]
            out.append(
                Reason(
                    code=grp.code,
                    title=grp.title,
                    feature=name,
                    label=self.labels[best],
                    value=raw_values.get(name),
                    typical=self.typical.get(name),
                    contribution=float(group_sum[g]),
                )
            )
        return out
