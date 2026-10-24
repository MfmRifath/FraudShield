"""FastAPI application.

Run: ``uvicorn --factory fraudshield.serving.app:create_app`` (see the Dockerfile).
Tests pass their own components to ``create_app`` instead of the configured ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudshield.cases.postgres import PostgresCaseStore
from fraudshield.cases.store import (
    AuditEntry,
    Case,
    CaseNotFoundError,
    CaseStore,
    InMemoryCaseStore,
)
from fraudshield.config import Settings, get_settings
from fraudshield.features.online import OnlineSpec
from fraudshield.features.redis_store import RedisFeatureStore
from fraudshield.features.store import FeatureStore, InMemoryFeatureStore
from fraudshield.monitoring.summary import counter_by, histogram_quantiles, latest_drift
from fraudshield.rules import RuleSet
from fraudshield.serving.activity import ActivityLog, InMemoryActivityLog, RedisActivityLog
from fraudshield.serving.breaker import CircuitBreaker
from fraudshield.serving.events import Broadcaster, MemorySink, Publisher, Sink
from fraudshield.serving.kafka_sink import KafkaSink
from fraudshield.serving.metrics import ServingMetrics
from fraudshield.serving.models import ModelLoadError, ModelManager
from fraudshield.serving.schemas import (
    CanaryRequest,
    CaseContext,
    LabelRequest,
    ModelInfo,
    PolicyRequest,
    ScoreRequest,
    ScoreResponse,
)
from fraudshield.serving.scorer import ReviewCapacity, ScoringService
from fraudshield.streaming.common import decode_row
from fraudshield.training.costs import COST_FIELDS, CostConfig, policy_outcome

log = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    service: ScoringService
    models: ModelManager
    cases: CaseStore
    metrics: ServingMetrics
    load_error: str | None = None
    activity: ActivityLog | None = None


def _state(request: Request) -> AppState:
    st: AppState = request.app.state.fs
    return st


StateDep = Annotated[AppState, Depends(_state)]


def create_app(
    settings: Settings | None = None,
    *,
    models: ModelManager | None = None,
    store: FeatureStore | None = None,
    sink: Sink | None = None,
    cases: CaseStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    srv = settings.serving

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        metrics = ServingMetrics()
        mm = models or ModelManager(settings)
        load_error = None
        if mm.champion is None:
            try:
                await asyncio.to_thread(mm.reload)
            except (
                ModelLoadError,
                OSError,
            ) as e:  # stay up (not ready) so /health and /metrics work
                load_error = str(e)
                log.error("model load failed: %s", e)
        metrics.set_model_versions(mm.versions())
        spec = OnlineSpec.from_feature_set(mm.feature_set)
        fstore: FeatureStore
        if store is not None:
            fstore = store
        elif srv.feature_store == "redis":
            st_cfg = settings.streaming
            fstore = RedisFeatureStore.from_url(spec, st_cfg.redis_url, st_cfg.redis_timeout_s)
        else:
            fstore = InMemoryFeatureStore(spec)
        # Open the store's connection pool before serving: otherwise the first /score after a
        # (re)start pays connection setup inside its 10 ms feature budget and gets degraded.
        if not await fstore.ping():
            log.warning("feature store not reachable at startup; /score will use fallbacks")
        if sink is not None:
            out_sink: Sink = sink
        elif srv.publisher == "kafka":
            out_sink = KafkaSink(settings.streaming.bootstrap_servers)
        else:
            out_sink = MemorySink()
        publisher = Publisher(out_sink, srv.publish_queue_size,
                              on_drop=lambda t: metrics.publish_dropped.labels(topic=t).inc())  # fmt: skip
        publisher.start()
        case_store: CaseStore
        pg_store: PostgresCaseStore | None = None
        if cases is not None:
            case_store = cases
        elif settings.cases.store == "postgres":
            pg_store = await PostgresCaseStore.connect(
                settings.cases.dsn.get_secret_value(), settings.cases.pool_max
            )
            case_store = pg_store
        else:
            case_store = InMemoryCaseStore()
        activity: ActivityLog = (
            RedisActivityLog(fstore.client)
            if isinstance(fstore, RedisFeatureStore)
            else InMemoryActivityLog()
        )
        costs = CostConfig.load(settings.paths.costs)
        service = ScoringService(
            models=mm, store=fstore, rules=RuleSet.load(settings.paths.rules), costs=costs,
            publisher=publisher, broadcaster=Broadcaster(srv.sse_queue_size), cases=case_store,
            metrics=metrics, breaker=CircuitBreaker(srv.breaker_failure_threshold, srv.breaker_reset_s),
            feature_timeout_s=srv.feature_timeout_ms / 1e3, update_store_inline=srv.update_store_inline,
            capacity=ReviewCapacity(costs.review_capacity_per_day), activity=activity,
            write_activity=srv.update_store_inline,
        )  # fmt: skip
        app.state.fs = AppState(settings, service, mm, case_store, metrics, load_error, activity)
        try:
            yield
        finally:
            await publisher.close()
            if isinstance(fstore, RedisFeatureStore):
                await fstore.close()
            if pg_store is not None:
                await pg_store.close()

    app = FastAPI(title="FraudShield", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(st: StateDep) -> JSONResponse:
        ok = st.models.champion is not None
        body = {"ready": ok, "models": st.models.versions(), "error": st.load_error}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics(st: StateDep) -> Response:
        return Response(generate_latest(st.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/score")
    async def score(req: ScoreRequest, st: StateDep) -> ScoreResponse:
        return await st.service.score(req)

    @app.get("/model")
    async def model_info(st: StateDep) -> ModelInfo:
        v = st.models.versions()
        return ModelInfo(champion=v["champion"], challenger=v["challenger"],
                         canary_pct=st.models.canary_pct, dataset=st.settings.serving.dataset)  # fmt: skip

    def require_admin(st: AppState, x_admin_token: str | None) -> None:
        token = st.settings.serving.admin_token
        if token is None:
            raise HTTPException(403, "admin disabled: FRAUDSHIELD_SERVING__ADMIN_TOKEN not set")
        if x_admin_token is None or not secrets.compare_digest(
            x_admin_token, token.get_secret_value()
        ):
            raise HTTPException(401, "invalid admin token")

    @app.post("/admin/canary")
    async def set_canary(
        body: CanaryRequest, st: StateDep, x_admin_token: Annotated[str | None, Header()] = None
    ) -> ModelInfo:
        require_admin(st, x_admin_token)
        if body.pct > 0 and st.models.challenger is None:
            raise HTTPException(409, "no @challenger loaded: cannot route traffic to it")
        st.models.canary_pct = body.pct
        log.info("canary set to %d%%", body.pct)
        return await model_info(st)

    @app.post("/admin/reload")
    async def reload(
        st: StateDep, x_admin_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        require_admin(st, x_admin_token)
        try:
            versions = await asyncio.to_thread(st.models.reload)
        except (ModelLoadError, OSError) as e:
            raise HTTPException(409, f"reload failed, previous models kept: {e}") from e
        st.metrics.set_model_versions(versions)
        st.load_error = None
        return {"reloaded": True, "models": versions}

    @app.get("/events/stream", response_class=EventSourceResponse)
    async def events(request: Request, st: StateDep) -> AsyncIterable[ServerSentEvent]:
        q = st.service.broadcaster.subscribe()
        try:
            yield ServerSentEvent(comment="fraudshield decisions")
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                except TimeoutError:
                    continue
                yield ServerSentEvent(data=event, event="decision", id=str(event["txn_id"]))
        finally:
            st.service.broadcaster.unsubscribe(q)

    @app.get("/cases")
    async def list_cases(
        st: StateDep,
        status: Annotated[str | None, Query(pattern="^(open|labeled)$")] = "open",
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[Case]:
        return await st.cases.list_cases(status, limit)

    @app.get("/cases/{case_id}")
    async def get_case(case_id: str, st: StateDep) -> Case:
        try:
            return await st.cases.get(case_id)
        except CaseNotFoundError as e:
            raise HTTPException(404, f"case {case_id} not found") from e

    @app.post("/cases/{case_id}/label")
    async def label_case(case_id: str, body: LabelRequest, st: StateDep) -> Case:
        try:
            case = await st.cases.label(
                case_id, body.label, body.analyst_id, body.notes, datetime.now(UTC)
            )
        except CaseNotFoundError as e:
            raise HTTPException(404, f"case {case_id} not found") from e
        await st.service.record_label(case)
        st.metrics.review_queue_depth.set(await st.cases.open_count())
        return case

    @app.get("/cases/{case_id}/context")
    async def case_context(case_id: str, st: StateDep) -> CaseContext:
        """Card timeline (last N events) and entities linked through shared device / email."""
        try:
            case = await st.cases.get(case_id)
        except CaseNotFoundError as e:
            raise HTTPException(404, f"case {case_id} not found") from e
        row = decode_row(case.txn)
        uid = str(row.get("uid"))
        timeline = (
            await st.activity.recent(uid, st.settings.cases.timeline_events) if st.activity else []
        )
        linked: dict[str, list[str]] = {}
        available = True
        try:
            state = await asyncio.wait_for(st.service.store.fetch(row), timeout=1.0)
            for col in ("device_info", "p_email_domain"):
                hist = state.history.get(col, [])
                uids = sorted(
                    {
                        str(e.values.get("uid"))
                        for e in hist
                        if e.values.get("uid") not in (None, uid)
                    }
                )
                if row.get(col) is not None:
                    linked[col] = uids
        except Exception as e:  # the context is best-effort; the case itself is in the case store
            log.warning("case context: feature store unavailable: %s", e)
            available = False
        return CaseContext(
            case_id=case_id, uid=uid, timeline=timeline, linked=linked, store_available=available
        )

    @app.get("/cases/{case_id}/audit")
    async def case_audit(case_id: str, st: StateDep) -> list[AuditEntry]:
        return await st.cases.audit_trail(case_id)

    @app.get("/monitoring/summary")
    async def monitoring_summary(st: StateDep) -> dict[str, Any]:
        reg = st.metrics.registry
        return {
            "models": st.models.versions(),
            "canary_pct": st.models.canary_pct,
            "decisions": counter_by(reg, "decisions", "decision"),
            "fallbacks": {k: v for k, v in counter_by(reg, "fallback", "reason").items() if v > 0},
            "latency_s": histogram_quantiles(reg, "score_latency_seconds"),
            "review_queue_depth": await st.cases.open_count(),
            "drift": latest_drift(st.settings.monitoring.report_dir),
        }

    @app.get("/monitoring/drift/latest.html", response_class=FileResponse)
    async def drift_report(st: StateDep) -> FileResponse:
        report_dir = st.settings.monitoring.report_dir
        d = latest_drift(report_dir)
        if not d or not d.get("has_html"):
            raise HTTPException(404, "no drift report yet")
        latest = json.loads((report_dir / "latest.json").read_text())
        name = Path(str(latest["report_html"])).name  # basename only: no path traversal
        return FileResponse(report_dir / name, media_type="text/html")

    def _window(st: AppState) -> Any:
        champion = st.models.champion
        if champion is None or champion.bundle.policy_window is None:
            raise HTTPException(409, "the champion model has no policy window (retrain to add one)")
        return champion.bundle.policy_window

    def _window_info(w: Any) -> dict[str, int]:
        return {"rows": w.height, "days": w["day"].n_unique(), "frauds": int(w["label"].sum())}

    @app.get("/policy/current")
    async def policy_current(st: StateDep) -> dict[str, Any]:
        w = _window(st)
        costs = st.service.costs
        return {
            "costs": costs.model_dump(),
            "outcome": policy_outcome(w, costs),
            "window": _window_info(w),
        }

    @app.post("/policy/simulate")
    async def policy_simulate(body: PolicyRequest, st: StateDep) -> dict[str, Any]:
        """What-if: recompute decisions and dollars on the held-out window with other costs."""
        w = _window(st)
        current = st.service.costs
        update = {f: getattr(body, f) for f in COST_FIELDS if getattr(body, f) is not None}
        update["review_capacity_per_day"] = body.review_capacity_per_day
        simulated = current.model_copy(update=update)
        return {
            "window": _window_info(w),
            "current": {"costs": current.model_dump(), "outcome": policy_outcome(w, current)},
            "simulated": {"costs": simulated.model_dump(), "outcome": policy_outcome(w, simulated)},
        }

    return app
