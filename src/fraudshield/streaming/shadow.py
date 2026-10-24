"""Shadow scorer: re-scores every served decision with ``@challenger`` -> ``shadow.scores``.

It uses the features the champion actually saw (carried in the decision event), so the
comparison is like-for-like and never queries the feature store. Degraded decisions are shadowed
with the challenger's degraded booster.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fraudshield.config import Settings
from fraudshield.model_bundle import ModelBundle, Variant
from fraudshield.serving.models import ModelLoadError, ModelManager, ModelSlot
from fraudshield.streaming.common import WorkerMetrics, json_topic, make_app
from fraudshield.training.costs import CostConfig, decide_one

log = logging.getLogger(__name__)


def shadow_score(
    event: dict[str, Any], bundle: ModelBundle, version: str, costs: CostConfig
) -> dict[str, Any]:
    variant: Variant = "degraded" if event.get("degraded") else "full"
    p, _ = bundle.score_one(event["features"], variant, n_reasons=0)
    decision = decide_one(p, float(event["amount"]), costs)
    return {
        "txn_id": event["txn_id"],
        "uid": event["uid"],
        "event_time": event["event_time"],
        "champion": {
            "version": event.get("model_version"),
            "probability": event.get("probability"),
            "decision": event["decision"],
            "variant": event.get("variant"),
        },
        "challenger": {"version": version, "probability": p, "decision": decision},
        "agree": decision == event["decision"],
    }


def load_challenger(settings: Settings) -> ModelSlot | None:
    srv = settings.serving
    if settings.streaming.challenger_bundle_dir is not None:
        srv = srv.model_copy(
            update={"challenger_bundle_dir": settings.streaming.challenger_bundle_dir}
        )
    mm = ModelManager(
        settings.model_copy(update={"serving": srv.model_copy(update={"load_challenger": True})})
    )
    try:
        mm.reload()
    except ModelLoadError as e:
        log.warning("shadow: cannot load models yet: %s", e)
        return None
    return mm.challenger


def run(settings: Settings, retry_s: float = 60.0) -> None:
    st = settings.streaming
    metrics = WorkerMetrics("shadow")
    metrics.serve(st.metrics_port)
    costs = CostConfig.load(settings.paths.costs)
    slot = load_challenger(settings)
    while slot is None:
        time.sleep(retry_s)
        slot = load_challenger(settings)
    log.info("shadowing with challenger %s", slot.version)
    app = make_app(settings, "fraudshield-shadow")
    out = json_topic(app, settings, st.topic_shadow)

    def handle(value: dict[str, Any], key: Any, ts: int, headers: Any) -> dict[str, Any] | None:
        metrics.observe(st.topic_decisions, ts)
        if value.get("variant") == "rules_only" or not value.get("features"):
            return None  # no model output to compare against
        return shadow_score(value, slot.bundle, slot.version, costs)

    sdf = app.dataframe(json_topic(app, settings, st.topic_decisions)).apply(handle, metadata=True)
    sdf = sdf.filter(lambda v: v is not None)
    sdf.to_topic(out, key=lambda v: str(v["uid"]))
    app.run()
