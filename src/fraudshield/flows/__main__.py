"""``python -m fraudshield.flows {serve,run <flow>}``."""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from fraudshield.flows.flows import (
    backfill_features,
    build_training_set_flow,
    drift_flow,
    mlops_demo_flow,
    promote_flow,
    retrain_and_promote,
    retrain_flow,
)

FLOWS: dict[str, Any] = {
    "backfill-features": backfill_features,
    "build-training-set": build_training_set_flow,
    "retrain": retrain_flow,
    "promote": promote_flow,
    "retrain-and-promote": retrain_and_promote,
    "mlops-demo": mlops_demo_flow,
    "drift": drift_flow,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["serve"]:
        from prefect import serve
        from prefect.deployments.runner import RunnerDeployment

        from fraudshield.config import get_settings
        from fraudshield.monitoring.metrics import serve_drift_metrics

        s = get_settings()
        serve_drift_metrics(s.monitoring.report_dir, s.monitoring.metrics_port)
        deployments = [
            retrain_and_promote.to_deployment(name="weekly", cron="0 3 * * 1"),  # Mon 03:00 UTC
            drift_flow.to_deployment(name="daily", cron="30 2 * * *"),  # daily 02:30 UTC
            *(f.to_deployment(name="on-demand") for f in (retrain_flow, promote_flow,
              build_training_set_flow, backfill_features, mlops_demo_flow, drift_flow)),
        ]  # fmt: skip
        # to_deployment is sync/async-compatible; called synchronously it returns the deployment
        serve(*(cast(RunnerDeployment, d) for d in deployments))
        return 0
    if len(args) == 2 and args[0] == "run" and args[1] in FLOWS:
        print(json.dumps(FLOWS[args[1]](), indent=2, default=str))
        return 0
    print(
        f"usage: python -m fraudshield.flows {{serve | run {{{','.join(FLOWS)}}}}}", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
