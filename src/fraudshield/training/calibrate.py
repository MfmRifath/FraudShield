"""Probability calibration fitted on the validation window.

Isotonic regression by default; Platt scaling (logistic on the score's logit) when the
validation window has fewer than ``platt_below_positives`` frauds, where isotonic overfits.
The fitted calibrator is plain data (JSON) and is applied with numpy only, so serving needs
neither scikit-learn nor pickles.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]
_EPS: Final = 1e-6


def _logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, _EPS, 1 - _EPS)
    out: FloatArray = np.log(p / (1 - p))
    return out


class Calibrator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["isotonic", "platt", "identity"]
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    a: float = 1.0
    b: float = 0.0

    def apply(self, scores: npt.ArrayLike) -> FloatArray:
        s = np.asarray(scores, dtype=np.float64)
        if self.kind == "isotonic":
            # same as IsotonicRegression(out_of_bounds="clip").predict: linear interp, clipped ends
            out: FloatArray = np.interp(s, np.asarray(self.x), np.asarray(self.y))
            return out
        if self.kind == "platt":
            out = 1.0 / (1.0 + np.exp(-(self.a * _logit(s) + self.b)))
            return out
        return np.clip(s, 0.0, 1.0)


def fit_calibrator(
    scores: npt.ArrayLike, y: npt.ArrayLike, platt_below_positives: int = 1000
) -> Calibrator:
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(y, dtype=np.int64)
    n_pos = int(t.sum())
    if n_pos == 0 or n_pos == t.size:
        log.warning("calibration window has a single class (%d positives); using identity", n_pos)
        return Calibrator(kind="identity")
    if n_pos < platt_below_positives:
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(s).reshape(-1, 1), t)
        return Calibrator(kind="platt", a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(s, t)
    return Calibrator(
        kind="isotonic",
        x=tuple(float(v) for v in iso.X_thresholds_),
        y=tuple(float(v) for v in iso.y_thresholds_),
    )
