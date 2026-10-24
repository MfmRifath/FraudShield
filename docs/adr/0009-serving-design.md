# ADR-0009: Serving design (online features, fallbacks, packaging)

- Status: Accepted
- Date: 2026-10-24
- Phase: 3

## Context

`/score` must answer within a tight latency budget, compute the same features as training, degrade gracefully when the feature store or the model fails, and never return 5xx for internal faults.

## Decision

1. **Online executor = same specs, store holds state only.**
   - `features/online.py` derives an `OnlineSpec` from the `FeatureSet`: per-entity history fields and retention (the longest window), seen-sets for first-seen flags, last-event times, and target-encoding columns.
   - `compute_online(row, state, fs)` is a pure function of the current row plus the state of strictly-earlier events.
   - Stores (`InMemoryFeatureStore` now, `RedisFeatureStore` in Phase 4) only persist and return that state.
   - Parity with the offline executor (< 1e-6 per feature) is tested on the 2,000-event Sparkov fixture and on Hypothesis-generated frames. Planting an inclusive window edge or a same-day label leak in the online executor makes the parity tests fail.
2. **Target encodings online:** labels go into a heap keyed by their visible snapshot (next 00:00 UTC at or after `label_available_time`) and are folded into cumulative counts as snapshots advance. This is the same daily-snapshot semantics as offline (ADR-0005).
3. **Pipeline** (`serving/scorer.py`), in order:
   1. validate (pydantic, 422 on bad input);
   2. feature fetch under `asyncio.wait_for(10 ms)` and a circuit breaker (5 consecutive failures, then open for 30 s, then one half-open trial);
   3. online features;
   4. hard rules;
   5. model (one booster, `num_threads=1`) + calibrator;
   6. cost policy;
   7. reason codes;
   8. side effects (publish, SSE, case, store update), which never change the answer.
4. **Fallbacks:** store failure/timeout/open breaker → degraded request-only booster from the same bundle. Model error or no model → rules-only (probability `null`, `variant="rules_only"`). Any unexpected exception → `decision="review"`, `fallback="internal_error"`: a human decides when the system's state is unknown. Every fallback increments `fallback_total{reason}`.
5. **Combining rules and policy:** the final decision is the more severe of the policy decision and the rule action.
   - Review capacity is a per-UTC-day counter, so online it is first-come-first-served. The offline evaluation keeps the highest-saving reviews of the day instead, so online savings can be somewhat lower than offline.
   - Over-capacity rule reviews become approve (as in the rules-only baseline); over-capacity model reviews take the cheaper of approve/block.
   - The counter is per process; a shared counter (Redis) can come later if the API is scaled out.
6. **Response:** spec fields (`txn_id, probability, decision, reason_codes[≤3], model_version, degraded, latency_ms`) plus `variant`, `rules_fired` and `fallback` for operability. Fired rules come first in `reason_codes`, then model reasons.
7. **Models:**
   - `model_source=bundle` loads a bundle directory (demo, tests).
   - `model_source=mlflow` resolves `models:/fraudshield@champion` (and `@challenger`), downloads the pyfunc and loads its bundle artifact directly (no pyfunc overhead).
   - A model whose `feature_set_hash` differs from what serving computes is refused (train/serve skew guard).
   - Hot reload (`POST /admin/reload`, `X-Admin-Token`, constant-time compare) builds the new slots first and swaps one reference; on failure the old models keep serving (409).
   - Canary: `blake2b(uid) % 100 < canary_pct` → `@challenger`.
8. **Events:** a bounded `asyncio.Queue` drained by a background task into a `Sink`. A full queue drops the event and counts it in `publish_dropped_total`, so `/score` never waits on Kafka. SSE uses FastAPI's native `EventSourceResponse` with bounded per-client queues (slow clients lose the oldest events).
9. **Dependencies split:** core (`[project.dependencies]`) = data + serving, including `mlflow-skinny`. Training libraries are in the `train` group. The Docker image installs core only; `ORJSONResponse` is deprecated in FastAPI, so responses use Pydantic serialisation and orjson is used for event payloads.
10. **Latency benchmark** (`scripts/bench_serving.py`):
    - uses a synthetic bundle of production-like size (500 trees, mean leaves/tree reported);
    - trains it with 30 % label noise, because otherwise the easy planted signal produced stumps (2.6 leaves/tree) and an optimistic benchmark;
    - writes `reports/benchmarks/serving_inprocess.md`.

## Consequences

- The API image is free of scikit-learn, optuna, pandas and matplotlib; training images add the `train` group (Phase 5).
- Starlette 1.9's TestClient expects `httpx2`, so it replaces `httpx` in the dev group.
- Cases are in memory until the Postgres store (Phase 7). The API updates the in-memory store inline (`update_store_inline=true`); with the stream processor (Phase 4) this is switched off.
