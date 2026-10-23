"""MLflow packaging: the model bundle as a pyfunc, registered with a ``@challenger`` alias.

MLflow model stages are deprecated; promotion moves the ``champion`` / ``challenger`` aliases.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient

from fraudshield.model_bundle import ModelBundle
from fraudshield.training.costs import decide

_REQUIREMENTS = ("lightgbm", "numpy", "pandas", "polars", "pyarrow", "pydantic", "mlflow")


class FraudShieldPyfunc(PythonModel):
    """``predict`` takes a feature frame; returns probability + policy decision (no capacity).

    ``params={"variant": "degraded"}`` scores with the request-only model.
    """

    def load_context(self, context: Any) -> None:
        self.bundle = ModelBundle.load(Path(context.artifacts["bundle"]))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        variant = (params or {}).get("variant", "full")
        df = pl.from_pandas(model_input)
        p = self.bundle.predict_proba(df, variant)
        costs = self.bundle.spec.costs.model_copy(update={"review_capacity_per_day": None})
        decisions = decide(p, df["amount"].to_numpy(), np.zeros(df.height), costs)
        return pd.DataFrame({"probability": p, "decision": decisions.astype(object)})


def log_and_register(
    bundle_dir: Path,
    example: pl.DataFrame,
    registered_model: str,
    alias: str = "challenger",
) -> tuple[str, str]:
    """Log the bundle as a pyfunc in the active run, register it, set ``alias``.

    Returns (model_uri, registered version).
    """
    bundle = ModelBundle.load(bundle_dir)
    cols = list(dict.fromkeys([*bundle.spec.full.feature_names, "amount"]))
    example_pd = example.select(cols).head(5).to_pandas()
    pyfunc = FraudShieldPyfunc()
    pyfunc.bundle = bundle
    signature = infer_signature(
        example_pd, pyfunc.predict(None, example_pd), params={"variant": "full"}
    )
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=pyfunc,
        artifacts={"bundle": str(bundle_dir)},
        signature=signature,
        input_example=example_pd,
        registered_model_name=registered_model,
        pip_requirements=[f"{p}=={version(p)}" for p in _REQUIREMENTS],
    )
    ver = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(registered_model, alias, ver)
    return info.model_uri, ver
