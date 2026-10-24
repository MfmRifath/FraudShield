"""The /score pipeline (ADR-0009).

validate -> feature fetch (timeout + circuit breaker) -> online features -> hard rules ->
LightGBM (num_threads=1) -> calibrator -> cost policy (+ daily review capacity) ->
reason codes -> non-blocking publish + SSE + review case.

Fallbacks: feature store failure -> degraded (request-only) model -> rules-only;
model error -> rules-only. ``score`` never raises: internal faults become a rules-only answer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from fraudshield.cases.store import Case, CaseStore
from fraudshield.features.online import EntityState, compute_online
from fraudshield.features.store import FeatureStore, FeatureStoreError
from fraudshield.model_bundle import Variant as BundleVariant
from fraudshield.reasons import Reason
from fraudshield.rules import RuleResult, RuleSet
from fraudshield.schema import Decision
from fraudshield.serving.activity import ActivityLog
from fraudshield.serving.activity import summary as activity_summary
from fraudshield.serving.breaker import CircuitBreaker
from fraudshield.serving.events import Broadcaster, Publisher
from fraudshield.serving.metrics import ServingMetrics
from fraudshield.serving.models import ModelManager, ModelSlot
from fraudshield.serving.schemas import ReasonCode, ScoreRequest, ScoreResponse, Variant
from fraudshield.training.costs import CostConfig, decide_one, expected_costs

log = logging.getLogger(__name__)
_SEVERITY: dict[str, int] = {"approve": 0, "review": 1, "block": 2}
DECISIONS_TOPIC = "decisions"
LABELS_TOPIC = "labels"


class ReviewCapacity:
    """Per-UTC-day review counter. In-process: one counter per API replica (see ADR-0009)."""

    def __init__(self, per_day: int | None) -> None:
        self.per_day = per_day
        self._day: str | None = None
        self._used = 0

    def try_acquire(self, t: datetime) -> bool:
        if self.per_day is None:
            return True
        day = t.astimezone(UTC).date().isoformat()
        if day != self._day:
            self._day, self._used = day, 0
        if self._used >= self.per_day:
            return False
        self._used += 1
        return True


@dataclass
class ScoringService:
    models: ModelManager
    store: FeatureStore
    rules: RuleSet
    costs: CostConfig
    publisher: Publisher
    broadcaster: Broadcaster
    cases: CaseStore
    metrics: ServingMetrics
    breaker: CircuitBreaker
    feature_timeout_s: float
    update_store_inline: bool
    capacity: ReviewCapacity
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    activity: ActivityLog | None = None
    write_activity: bool = False  # in-process mode; with the stream processor it writes instead
    model_executor: Executor | None = None  # None: run the model on the event loop

    async def score(self, req: ScoreRequest, explain: bool = True) -> ScoreResponse:
        """``explain=False`` skips model reason codes (no ``pred_contrib`` pass)."""
        t0 = time.perf_counter()
        try:
            return await self._score(req, t0, explain)
        except Exception:  # last line of defence: never a 5xx for internal faults
            log.exception("unexpected error scoring %s", req.txn_id)
            self.metrics.fallbacks.labels(reason="internal_error").inc()
            return ScoreResponse(
                # unknown state: neither approve (fraud risk) nor block (customer harm); a human decides
                txn_id=req.txn_id, probability=None, decision="review",
                reason_codes=[ReasonCode(code="INTERNAL", message="scoring error; routed for review")],
                model_version=None, degraded=True, latency_ms=(time.perf_counter() - t0) * 1e3,
                variant="rules_only", fallback="internal_error",
            )  # fmt: skip

    async def _fetch(self, row: dict[str, Any]) -> tuple[EntityState | None, str | None]:
        if not self.breaker.allow():
            return None, "circuit_open"
        t = time.perf_counter()
        try:
            state = await asyncio.wait_for(self.store.fetch(row), timeout=self.feature_timeout_s)
        except TimeoutError:
            self.breaker.failure()
            return None, "feature_timeout"
        except Exception as e:  # FeatureStoreError or a client library error
            self.breaker.failure()
            log.debug("feature fetch failed: %s", e)  # counted in fallback_total; avoid log floods
            return None, "feature_store_error"
        except BaseException:  # cancelled (client gone, shutdown): no verdict on the store
            self.breaker.abandon()
            raise
        finally:
            self.metrics.feature_fetch.observe(time.perf_counter() - t)
        self.breaker.success()
        return state, None

    def _features(self, row: dict[str, Any], state: EntityState | None) -> dict[str, Any]:
        fs = self.models.feature_set
        if state is not None:
            return compute_online(row, state, fs)
        feats: dict[str, Any] = dict.fromkeys(fs.names)
        feats.update({f.name: f.scalar(row) for f in fs.request_features})
        return feats

    async def _score(self, req: ScoreRequest, t0: float, explain: bool) -> ScoreResponse:
        now = self.clock()
        row = req.to_row(now)
        slot = self.models.route(req.uid)
        state, fallback = await self._fetch(row)
        feats = self._features(row, state)
        rule = self.rules.evaluate_row(feats)

        probability: float | None = None
        reasons: list[Reason] = []
        variant: Variant = "rules_only"
        model_version: str | None = None
        policy: Decision = "approve"
        if slot is None:
            fallback = fallback or "no_model"
        else:
            try:
                which: BundleVariant = "full" if state is not None else "degraded"
                n_reasons = 3 if explain else 0
                if self.model_executor is None:
                    p, reasons = slot.bundle.score_one(feats, which, n_reasons)
                else:  # LightGBM releases the GIL: the loop keeps serving other requests
                    p, reasons = await asyncio.get_running_loop().run_in_executor(
                        self.model_executor, slot.bundle.score_one, feats, which, n_reasons
                    )
                probability = p
                policy = decide_one(p, req.amount, self.costs)
                variant = "champion" if slot.alias == "champion" else "challenger"
                model_version = slot.version
            except Exception:
                log.exception("model error for %s", req.txn_id)
                fallback = "model_error"
                probability, reasons = None, []

        decision = self._combine(policy, rule)
        if decision == "review" and not self.capacity.try_acquire(row["event_time"]):
            decision = self._over_capacity(probability, req.amount, rule)
            self.metrics.fallbacks.labels(reason="review_capacity").inc()

        degraded = state is None or variant == "rules_only"
        if fallback:
            self.metrics.fallbacks.labels(reason=fallback).inc()
        resp = ScoreResponse(
            txn_id=req.txn_id,
            probability=probability,
            decision=decision,
            reason_codes=self._reason_codes(rule, reasons),
            model_version=model_version,
            degraded=degraded,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            variant=variant,
            rules_fired=list(rule.fired),
            fallback=fallback,
        )
        self._observe(resp)
        await self._after(row, feats, resp, slot, state is not None)
        return resp

    @staticmethod
    def _combine(policy: Decision, rule: RuleResult) -> Decision:
        if rule.action is not None and _SEVERITY[rule.action] > _SEVERITY[policy]:
            return rule.action
        return policy

    def _over_capacity(self, p: float | None, amount: float, rule: RuleResult) -> Decision:
        if p is None or rule.action == "review":  # mirror the rules-only baseline: approve
            return "approve"
        ec = expected_costs(np.array([p]), np.array([amount]), self.costs)[0]
        return "approve" if ec[0] <= ec[2] else "block"

    @staticmethod
    def _reason_codes(rule: RuleResult, reasons: list[Reason]) -> list[ReasonCode]:
        codes = [
            ReasonCode(code=f"RULE:{n}", message=r)
            for n, r in zip(rule.fired, rule.reasons, strict=True)
        ]
        codes += [
            ReasonCode(
                code=r.code, message=r.message, feature=r.feature, value=r.value, typical=r.typical
            )
            for r in reasons
        ]
        return codes[:3]

    def _observe(self, resp: ScoreResponse) -> None:
        m = self.metrics
        m.score_latency.labels(
            model_version=resp.model_version or "none", degraded=str(resp.degraded).lower()
        ).observe(resp.latency_ms / 1e3)
        m.decisions.labels(decision=resp.decision).inc()
        if resp.probability is not None:
            m.fraud_score.observe(resp.probability)

    async def _after(
        self,
        row: dict[str, Any],
        feats: dict[str, Any],
        resp: ScoreResponse,
        slot: ModelSlot | None,
        fetched: bool,
    ) -> None:
        """Side effects that must not affect the answer: publish, SSE, case, store update."""
        event = {
            "txn_id": resp.txn_id, "uid": row["uid"], "event_time": row["event_time"],
            "amount": row["amount"], "decision": resp.decision, "probability": resp.probability,
            "model_version": resp.model_version, "variant": resp.variant, "degraded": resp.degraded,
            "latency_ms": resp.latency_ms, "rules_fired": resp.rules_fired,
            "reason_codes": [r.model_dump() for r in resp.reason_codes],
        }  # fmt: skip
        try:
            self.broadcaster.publish(event)
            self.publisher.publish(
                DECISIONS_TOPIC, row["uid"], event | {"row": row, "features": feats}
            )
            if resp.decision == "review":
                loss = (resp.probability if resp.probability is not None else 1.0) * (
                    row["amount"] + self.costs.chargeback_fee
                )
                await self.cases.open_case(
                    Case(
                        case_id=resp.txn_id,
                        created_at=self.clock(),
                        txn=row,
                        probability=resp.probability,
                        decision=resp.decision,
                        reasons=event["reason_codes"],
                        expected_loss=loss,
                        model_version=resp.model_version,
                    )
                )
                self.metrics.review_queue_depth.set(await self.cases.open_count())
            if self.update_store_inline and fetched:
                await self.store.update(row)
            if self.write_activity and self.activity is not None:
                await self.activity.append(row["uid"], activity_summary(event, row))
        except FeatureStoreError as e:
            log.warning("store update failed for %s: %s", resp.txn_id, e)
        except Exception:
            log.exception("post-score side effect failed for %s", resp.txn_id)

    async def record_label(self, case: Case) -> None:
        """Analyst label -> ``labels`` topic (Parquet sink, audit).

        It does not touch online target encodings: offline features count matured chargebacks
        only, and online must match them (parity); analyst labels reach models through
        ``build_training_set`` instead (ADR-0013).
        """
        if case.label is None or case.labeled_at is None:
            raise ValueError(f"case {case.case_id} is not labelled")
        self.publisher.publish(
            LABELS_TOPIC, case.txn["uid"],
            {"txn_id": case.case_id, "label": 1 if case.label == "fraud" else 0, "label_source": "analyst",
             "label_time": case.labeled_at, "analyst_id": case.analyst_id},
        )  # fmt: skip
