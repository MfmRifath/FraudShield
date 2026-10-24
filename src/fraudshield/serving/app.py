"""FastAPI application.

Run: ``uvicorn --factory fraudshield.serving.app:create_app`` (see the Dockerfile).
Tests pass their own components to ``create_app`` instead of the configured ones.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudshield.cases.store import Case, CaseNotFoundError, CaseStore, InMemoryCaseStore
from fraudshield.config import Settings, get_settings
from fraudshield.features.online import OnlineSpec
from fraudshield.features.store import FeatureStore, InMemoryFeatureStore
from fraudshield.rules import RuleSet
from fraudshield.serving.breaker import CircuitBreaker
from fraudshield.serving.events import Broadcaster, MemorySink, Publisher, Sink
from fraudshield.serving.metrics import ServingMetrics
from fraudshield.serving.models import ModelLoadError, ModelManager
from fraudshield.serving.schemas import LabelRequest, ModelInfo, ScoreRequest, ScoreResponse
from fraudshield.serving.scorer import ReviewCapacity, ScoringService
from fraudshield.training.costs import CostConfig

log = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    service: ScoringService
    models: ModelManager
    cases: CaseStore
    metrics: ServingMetrics
    load_error: str | None = None


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
        fstore = store or InMemoryFeatureStore(OnlineSpec.from_feature_set(mm.feature_set))
        publisher = Publisher(sink or MemorySink(), srv.publish_queue_size,
                              on_drop=lambda t: metrics.publish_dropped.labels(topic=t).inc())  # fmt: skip
        publisher.start()
        case_store = cases or InMemoryCaseStore()
        costs = CostConfig.load(settings.paths.costs)
        service = ScoringService(
            models=mm, store=fstore, rules=RuleSet.load(settings.paths.rules), costs=costs,
            publisher=publisher, broadcaster=Broadcaster(srv.sse_queue_size), cases=case_store,
            metrics=metrics, breaker=CircuitBreaker(srv.breaker_failure_threshold, srv.breaker_reset_s),
            feature_timeout_s=srv.feature_timeout_ms / 1e3, update_store_inline=srv.update_store_inline,
            capacity=ReviewCapacity(costs.review_capacity_per_day),
        )  # fmt: skip
        app.state.fs = AppState(settings, service, mm, case_store, metrics, load_error)
        try:
            yield
        finally:
            await publisher.close()

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
                         canary_pct=st.settings.serving.canary_pct, dataset=st.settings.serving.dataset)  # fmt: skip

    @app.post("/admin/reload")
    async def reload(
        st: StateDep, x_admin_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        token = st.settings.serving.admin_token
        if token is None:
            raise HTTPException(403, "reload disabled: FRAUDSHIELD_SERVING__ADMIN_TOKEN not set")
        if x_admin_token is None or not secrets.compare_digest(
            x_admin_token, token.get_secret_value()
        ):
            raise HTTPException(401, "invalid admin token")
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

    return app
