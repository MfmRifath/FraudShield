"""The deployable model artifact: two LightGBM boosters + everything needed to use them.

A bundle directory contains ``full.txt`` and ``degraded.txt`` (LightGBM text models) and
``bundle.json`` (feature orders, category vocabulary, calibrators, cost policy, typical values,
reason map, metadata). No pickles: it loads with lightgbm + numpy + polars only.

* ``full``: every feature; needs the feature store.
* ``degraded``: request-only features; used when the feature store is unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from fraudshield.reasons import Reason, ReasonMapper
from fraudshield.training.calibrate import Calibrator
from fraudshield.training.costs import CostConfig
from fraudshield.training.matrix import CategoryEncoder, build_matrix, build_row

BUNDLE_FORMAT: Final = 1
Variant = Literal["full", "degraded"]
FloatArray = npt.NDArray[np.float64]


def find_bundle_dir(model_dir: Path) -> Path:
    """Locate the bundle inside a downloaded MLflow pyfunc model directory.

    MLflow stores a pyfunc artifact under the *source directory's name* (``artifacts/<name>``),
    so the path is not fixed; search for ``bundle.json`` instead of assuming ``artifacts/bundle``.
    """
    found = sorted(model_dir.rglob("bundle.json"))
    if not found:
        raise FileNotFoundError(f"no bundle.json under {model_dir}")
    if len(found) > 1:
        raise ValueError(f"several bundles under {model_dir}: {found}")
    return found[0].parent


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VariantSpec(_Model):
    booster_file: str
    feature_names: tuple[str, ...]
    calibrator: Calibrator


class BundleSpec(_Model):
    format: int = BUNDLE_FORMAT
    metadata: dict[str, Any]
    encoder: CategoryEncoder
    full: VariantSpec
    degraded: VariantSpec
    costs: CostConfig
    typical: dict[str, float | str | None]
    reason_map: dict[str, dict[str, str]]


class ModelBundle:
    def __init__(self, spec: BundleSpec, boosters: Mapping[Variant, lgb.Booster]) -> None:
        if spec.format != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {spec.format}")
        self.spec = spec
        self.boosters = dict(boosters)
        # one mapper per variant: contribution vectors follow that variant's feature order
        self.reasons: dict[Variant, ReasonMapper] = {
            v: ReasonMapper.from_mapping(
                {n: spec.reason_map[n] for n in self._variant(v).feature_names}, spec.typical
            )
            for v in ("full", "degraded")
        }
        for v in ("full", "degraded"):
            names = self.boosters[v].feature_name()
            if list(names) != list(self._variant(v).feature_names):
                raise ValueError(f"{v} booster feature order does not match the bundle spec")

    def _variant(self, which: Variant) -> VariantSpec:
        return self.spec.full if which == "full" else self.spec.degraded

    # ------------------------------------------------------------------ io
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        for v in ("full", "degraded"):
            self.boosters[v].save_model(str(directory / self._variant(v).booster_file))
        (directory / "bundle.json").write_text(self.spec.model_dump_json(indent=2) + "\n")
        return directory

    @classmethod
    def load(cls, directory: Path) -> ModelBundle:
        spec = BundleSpec.model_validate_json((directory / "bundle.json").read_text())
        boosters: dict[Variant, lgb.Booster] = {
            "full": lgb.Booster(model_file=str(directory / spec.full.booster_file)),
            "degraded": lgb.Booster(model_file=str(directory / spec.degraded.booster_file)),
        }
        return cls(spec, boosters)

    # ------------------------------------------------------------------ scoring
    def matrix(self, df: pl.DataFrame, which: Variant = "full") -> FloatArray:
        return build_matrix(df, self._variant(which).feature_names, self.spec.encoder)

    def raw_scores(self, x: FloatArray, which: Variant = "full") -> FloatArray:
        out: FloatArray = np.asarray(
            self.boosters[which].predict(x, num_threads=1), dtype=np.float64
        )
        return out

    def predict_proba(self, df: pl.DataFrame, which: Variant = "full") -> FloatArray:
        """Calibrated P(fraud) for every row of a feature frame."""
        x = self.matrix(df, which)
        return self._variant(which).calibrator.apply(self.raw_scores(x, which))

    def contributions(self, x: FloatArray, which: Variant = "full") -> FloatArray:
        out: FloatArray = np.asarray(
            self.boosters[which].predict(x, pred_contrib=True, num_threads=1), dtype=np.float64
        )
        return out

    def score_one(
        self, features: Mapping[str, Any], which: Variant = "full", n_reasons: int = 3
    ) -> tuple[float, list[Reason]]:
        """Serving path for one event: calibrated probability + top reasons."""
        x = build_row(features, self._variant(which).feature_names, self.spec.encoder)
        p = float(self._variant(which).calibrator.apply(self.raw_scores(x, which))[0])
        if n_reasons <= 0:  # e.g. shadow scoring: skip the (costly) pred_contrib pass
            return p, []
        reasons = self.reasons[which].top(self.contributions(x, which)[0], features, n_reasons)
        return p, reasons
