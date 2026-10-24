# ADR-0011: MLOps loop (training set, registry history, promotion gate, Prefect)

- Status: Accepted
- Date: 2026-10-24
- Phase: 5

## Context

The spec asks for one command that runs retrain → challenger → shadow comparison → promote, a gate with four criteria, a rollback command, canary routing, and Prefect 3 flows on a compose `mlops` profile (Postgres, MLflow server, Prefect server + worker).

## Decision

1. **Training set** (`mlops/training_set.py`, pure):
   - Labels known as of a cutoff are the union of *matured* chargebacks (`label_available_time <= as_of`) and analyst labels (`label_time <= as_of`; the latest per transaction).
   - **The chargeback wins conflicts.** Only events up to the cutoff are eligible. Conflict and source counts are reported.
2. **Registry** (`mlops/registry.py`):
   - Aliases only.
   - Each promotion appends the previous champion to the `champion_history` registered-model tag; `rollback` pops it, so rollback is one command (`make rollback`).
   - Promotion and rollback timestamps go on version tags.
3. **Gate** (`mlops/gate.py`): every check must pass.
   - PR-AUC ≥ champion − 0.005, and net savings ≥ champion. Both are measured on the same test window with the same costs and capacity.
   - Single-row `score_one` p95 ≤ 5 ms.
   - Parity: the challenger's `feature_set_hash` equals what serving computes, **and** online/offline parity < 1e-6 on a canonical sample.

   The shadow comparison (decision agreement, mean |Δp|) is reported but not gating. `--force` exists for operators and is recorded in the report. Reports go to `reports/<ds>/promotions/*.md|json`.
4. **One command** (`make mlops-demo` / `fraudshield mlops-demo`), on synthetic data with a SQLite registry:
   1. bootstrap v1 → @champion;
   2. retrain v2 with analyst labels and different hyperparameters → @challenger;
   3. gate v2 against v1, with the shadow comparison;
   4. promote, or show the gate blocking v2, in which case a recorded forced promotion follows;
   5. rollback.
   - The demo adds 8 % label noise *before* feature computation. Without it the planted signal is perfectly separable, the two models come out identical, and the gate and shadow comparison would be vacuous.
   - On the current seed the gate **blocks** v2 (net savings 29,333 vs 29,518). The demo shows exactly that, then the forced promotion and rollback.
5. **Prefect 3.8.8**: thin flows over the plain functions. The worker container runs `python -m fraudshield.flows serve` (deployments, a weekly `retrain-and-promote` cron), so no separate work pool is needed. The same functions run from the CLI without Prefect (lite profile).
6. **FastAPI pinned to 0.150.0.** Prefect 3.8.8's server code uses `Body(min_items=...)`, which FastAPI removed in 0.155 (ephemeral flow runs crashed). Prefect 3.9 itself pins `fastapi<0.151`. Nothing we use from FastAPI changed between 0.150 and 0.162; the full suite passes.
7. **MLflow server**:
   - Postgres backend; proxied artifacts (`--serve-artifacts`), so clients need no shared filesystem. Experiments created against a server get no client-side artifact location.
   - `--allowed-hosts` includes the compose hostname (MLflow 3's DNS-rebinding guard rejects `mlflow:5000` otherwise).
   - `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`: the default GenAI job consumers took about 1 GB; without them the server uses 430 MiB.
   - The upstream image gets `psycopg2-binary` in a two-line Dockerfile.
8. **Canary toggle at runtime:** `POST /admin/canary {"pct": n}` (token-protected; 409 if no challenger is loaded) and `fraudshield canary n`.
9. **Two latent bugs this phase exposed, both fixed and regression-tested:**
   - Loading a model from the MLflow registry assumed `artifacts/bundle`, but MLflow stores a pyfunc artifact under the *source directory's* name. The loader now locates `bundle.json`.
   - `models:/` downloads resolved the registry via MLflow's process-global URI. `mlflow_utils` now pins both the tracking and registry URIs.
10. **Cold start:** the API pings the feature store during startup. The first request after a restart used to pay Redis connection setup inside its 10 ms budget and fall back to the degraded model (found by the integration test).

## Consequences

- `make up-mlops` + `make test-integration-mlops` verify the loop against the real MLflow server and as a Prefect deployment run (8/8 integration tests).
- Analyst labels come from a Parquet export until the Postgres case store (Phase 7) replaces it.
- Under fixed split windows, recent analyst labels land in the test window. The training-set builder supports rolling cutoffs; rolling retrain windows are a later improvement.
