"""MLflow access that never depends on process-global tracking/registry URIs.

``mlflow.artifacts.download_artifacts("models:/name/v", tracking_uri=X)`` resolves the
registry via the *global* registry URI, so a process that touched another tracking store can
load a model from the wrong registry. These helpers pin both URIs explicitly.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def download_model_version(tracking_uri: str, name: str, version: str) -> Path:
    """Download a registered model version's files and return the local directory."""
    source = client(tracking_uri).get_model_version_download_uri(name, version)
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=source, tracking_uri=tracking_uri))
