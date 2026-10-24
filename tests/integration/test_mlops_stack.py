"""MLOps profile end to end (``make test-integration-mlops``).

* the loop against the real MLflow server (Postgres backend, proxied artifacts);
* the same loop as a Prefect deployment run executed by the prefect-worker container.
"""

from __future__ import annotations

import os
import urllib.request
import uuid
from pathlib import Path
from typing import cast

import pytest
from mlflow.tracking import MlflowClient

from fraudshield.config import Settings
from fraudshield.mlops.demo import mlops_demo

pytestmark = [pytest.mark.integration, pytest.mark.mlops]
MLFLOW = "http://mlflow:5000"


def test_services_healthy() -> None:
    for url in (f"{MLFLOW}/health", "http://prefect-server:4200/api/health"):
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 - compose-internal
            assert r.status == 200, url


def test_loop_against_mlflow_server(tmp_path: Path) -> None:
    name = f"it-demo-{uuid.uuid4().hex[:8]}"
    out = mlops_demo(Settings(), tmp_path, tracking_uri=MLFLOW, model_name=name)
    client = MlflowClient(tracking_uri=MLFLOW)
    champion = client.get_model_version_by_alias(name, "champion")
    assert str(champion.version) == out["bootstrap"]["version"] == out["champion_after_rollback"]
    # artifacts were stored server-side (proxied), not in this container's filesystem
    source = client.get_model_version(name, out["challenger"]["version"]).source or ""
    assert source.startswith(("models:/", "mlflow-artifacts:")), source


def test_loop_as_prefect_deployment_run() -> None:
    os.environ["PREFECT_API_URL"] = "http://prefect-server:4200/api"
    from prefect.client.schemas.objects import FlowRun, StateType
    from prefect.deployments import run_deployment

    name = f"it-flow-{uuid.uuid4().hex[:8]}"
    params = {"workdir": f"artifacts/{name}", "model_name": name}
    # run_deployment is sync/async-compatible; called synchronously it returns the FlowRun
    run = cast(FlowRun, run_deployment("mlops-demo/on-demand", parameters=params, timeout=900))
    assert run.state is not None
    assert run.state.type == StateType.COMPLETED, run.state
    client = MlflowClient(tracking_uri=MLFLOW)
    assert client.get_model_version_by_alias(name, "champion").version is not None
