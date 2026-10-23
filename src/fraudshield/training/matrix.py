"""Feature frame -> numeric model matrix.

Categorical features become integer codes from a vocabulary fitted on the training window
(values seen at least ``min_count`` times, most frequent first). Null and unseen values become
NaN, which LightGBM treats as missing. The vocabulary is plain JSON and ships in the model
bundle, so serving encodes exactly like training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

FloatArray = npt.NDArray[np.float64]


class CategoryEncoder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vocab: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, df: pl.DataFrame, columns: Sequence[str], min_count: int) -> CategoryEncoder:
        vocab: dict[str, tuple[str, ...]] = {}
        for c in columns:
            counts = (
                df.select(pl.col(c).drop_nulls().cast(pl.String))
                .group_by(c)
                .len()
                .filter(pl.col("len") >= min_count)
                .sort(["len", c], descending=[True, False])
            )
            vocab[c] = tuple(counts[c].to_list())
        return cls(vocab=vocab)

    def expr(self, column: str) -> pl.Expr:
        values = self.vocab[column]
        return (
            pl.col(column)
            .cast(pl.String)
            .replace_strict(
                list(values), list(range(len(values))), default=None, return_dtype=pl.Float64
            )
            .alias(column)
        )

    def code(self, column: str, value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(self.vocab[column].index(str(value)))
        except ValueError:
            return float("nan")


def build_matrix(
    df: pl.DataFrame, feature_names: Sequence[str], encoder: CategoryEncoder
) -> FloatArray:
    """(n, k) float64 matrix in ``feature_names`` order, NaN for missing."""
    exprs = [
        encoder.expr(n) if n in encoder.vocab else pl.col(n).cast(pl.Float64).alias(n)
        for n in feature_names
    ]
    out: FloatArray = df.select(exprs).fill_null(float("nan")).to_numpy().astype(np.float64)
    return out


def build_row(
    features: Mapping[str, Any], feature_names: Sequence[str], encoder: CategoryEncoder
) -> FloatArray:
    """(1, k) matrix for a single feature dict (serving path; same encoding as ``build_matrix``)."""
    vals = []
    for n in feature_names:
        v = features.get(n)
        if n in encoder.vocab:
            vals.append(encoder.code(n, v))
        else:
            vals.append(float("nan") if v is None else float(v))
    return np.asarray([vals], dtype=np.float64)


def typical_values(
    df: pl.DataFrame, feature_names: Sequence[str], categorical: Sequence[str]
) -> dict[str, float | str | None]:
    """Median (numeric) or mode (categorical) per feature: the "typical" in reason codes."""
    out: dict[str, float | str | None] = {}
    for n in feature_names:
        col = df[n].drop_nulls()
        if col.is_empty():
            out[n] = None
        elif n in categorical:
            out[n] = str(col.mode().sort()[0])
        else:
            med = col.cast(pl.Float64).median()
            out[n] = None if med is None else float(med)  # type: ignore[arg-type]
    return out
