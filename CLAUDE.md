# CLAUDE.md: FraudShield project memory

> Read this file and PLAN.md before writing code in any session. Keep it under ~200 lines.

## Current phase

**Phase 7: Analyst UI: complete, awaiting review.** Next: Phase 8 (Hardening & deploy), only after the user types "continue". Real-data DoD items are deferred (rule 9).
IEEE-CIS raw data is **not downloaded yet** (needs the user's Kaggle token + accepted competition rules). Sparkov was downloaded and built in Phase 1.

## Goal

A recruiter-grade, production-style **real-time fraud detection service**: PIT-correct features, calibrated LightGBM + cost-aware policy, streaming feature updates, a champion/challenger MLOps loop, drift monitoring, an analyst UI. Runs on a 16 GB laptop (8 GB lite profile; < 1 GB demo container), arm64 + amd64.

## Working rules (from the user, non-negotiable)

1. Work **phase by phase**. At the end of each phase: full tests + linters, update PLAN.md (progress log) and this file, write a phase summary (built / how to run / test results / known issues / decisions needing input), then **STOP until the user types "continue"**.
2. Small conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `perf:`, `refactor:`). Never commit `data/raw`, secrets or large artifacts.
3. All business logic in `src/fraudshield`. Notebooks only import from the package.
4. Type hints everywhere; ruff + mypy `--strict` clean; pytest for everything; prefer pure functions.
5. Before adding/upgrading a dependency, check its changelog, then pin it exactly (ADR-0002).
6. For ambiguities, choose the simplest spec-consistent option, write an ADR, list it in the phase summary. Ask only when costly to reverse.
7. **Never fabricate metrics.** Every README number must come from a rerunnable script writing to `reports/`.
8. Tick PLAN.md items only when tests prove them.
9. **Code first, run later (user, 2026-10-23):** no dataset downloads, real training/tuning, data-driven reports or long jobs until *all* phases are coded. Verify against `tests/fixtures/` and synthetic data. Real-data steps live in PLAN.md "Deferred real-data runs" (done with the user at the end); DoD items needing real data are marked *deferred*, not ticked.

## Architecture summary

```
adapters (IEEE-CIS, Sparkov) -> canonical Transaction -> features/definitions.py (declarative specs)
   ├─ offline.py (Polars/DuckDB, PIT) -> training/* -> ModelBundle -> MLflow (@challenger)
   └─ online.py (pure) + FeatureStore (InMemory | Redis) -> serving/ (FastAPI /score)
streaming: replayer -> transactions.raw -> gateway -> /score -> decisions -> processor (Redis), parquet sink, shadow
           replayer -> transactions.truth -> label_emitter -> labels -> processor (TE counts chargebacks only)
cases:     /score review -> case store (InMemory | Postgres cases/labels/audit_log) -> UI label -> labels table + topic -> build_training_set
ui:        React (ui/) -> nginx :5180 -> /api -> FastAPI (SSE /events/stream, /cases/*, /monitoring/summary, /policy/*)
mlops:     build_training_set (chargeback ∪ analyst, PIT) -> retrain -> @challenger -> gate (+shadow) -> @champion; rollback
flows:     Prefect 3: backfill-features, build-training-set, retrain, promote, retrain-and-promote (weekly), drift (daily), mlops-demo
monitor:   parquet decisions/labels log -> drift flow (Evidently + PSI + matured PR-AUC) -> latest.json -> worker /metrics -> Prometheus alerts, Grafana
```

- Score path: validate -> feature fetch (10 ms timeout + circuit breaker) -> hard rules -> LightGBM (num_threads=1) -> calibrator -> cost policy (+ daily review capacity) -> top-3 reasons -> non-blocking publish.
- Fallbacks: store failure -> degraded request-only booster -> rules-only; unknown error -> `review`. Never 5xx for internal faults.
- Data: IEEE-CIS event_time = 2017-12-01 + TransactionDT s; uid = `card1_addr1_D1n` (D1n = floor(DT/86400) - D1); Sparkov uid = cc_num. IEEE splits (days): train 1-120, val 128-150, test 158-183.

## Repository layout (non-obvious parts)

```
src/fraudshield/config.py schema.py labels.py manifest.py pipeline.py cli.py eda.py rules.py reasons.py
                model_bundle.py (full.txt + degraded.txt + bundle.json; find_bundle_dir) mlflow_utils.py synthetic.py devtools.py
src/fraudshield/features/   definitions offline online store(InMemory, replay) redis_store parity
src/fraudshield/training/   splits matrix models calibrate costs evaluate tune package model_card train bundle_builder
src/fraudshield/serving/    app scorer models(slots, canary, reload) breaker events kafka_sink metrics schemas activity(card timeline)
src/fraudshield/streaming/  replayer gateway processor label_emitter parquet_sink shadow common; `python -m fraudshield.streaming <worker>`
src/fraudshield/mlops/      training_set registry(alias history) gate loop(retrain/promote/rollback) demo
src/fraudshield/flows/      Prefect flows; `python -m fraudshield.flows {serve | run <flow>}` (serve also exposes drift metrics on :9100)
src/fraudshield/monitoring/ drift (detect_drift, psi, top_features) metrics (DriftFileCollector) summary (UI: quantiles, counters, latest drift)
src/fraudshield/cases/      store (Case, CaseStore protocol, InMemory) postgres (PostgresCaseStore, fetch_analyst_labels)
ui/                         Vite + React app (src/pages, src/lib pure helpers, src/test vitest), Dockerfile + nginx.conf.template
configs/ base.yaml costs.yaml rules.yaml tuned/<ds>.json      data/manifests/ (COMMITTED)   reports/ (md/json/png committed)
infra/prometheus/{prometheus.yml,rules/}  infra/grafana/{provisioning,dashboards (generated by scripts/build_dashboards.py)}
compose.yml (profiles core, mlops, monitoring, ui)  Dockerfile (targets: runtime=API [default], trainer=Prefect worker)  infra/{dev,mlflow,postgres}
tests/ unit parity integration fixtures/sparkov_sample.csv (2,000 CC0 rows) oracle.py strategies.py serving_support.py case_contract.py
```

## Conventions

- **Python 3.12**, src layout, hatchling, `uv` everywhere (`uv run --frozen`). Absolute imports only. Line length 100, `ruff format`. Avoid `# fmt: skip` inside expressions (RUF028).
- **Config:** `get_settings()`; precedence init > env (`FRAUDSHIELD_*`, `__` nesting) > YAML (`FRAUDSHIELD_CONFIG`) > defaults; `extra="forbid"`, frozen. New profiles -> `PROFILES` in `tests/unit/test_config.py` (ADR-0003).
- **Datetimes** tz-aware UTC (ruff DTZ). **Canonical frame**: `CANONICAL_COLUMNS` then `x_*` Float64; `event_time` strictly increasing + unique (µs tie-break, ADR-0004).
- **Feature semantics** are documented once in `features/definitions.py`: windows `(t-w, t]` incl. current event; TE from daily 00:00 UTC snapshot (ADR-0005). New history features must work in `compute_online` and pass `tests/parity` (< 1e-6) on InMemory **and** Redis stores.
- **Polars 2** (ADR-0007): eager / `collect(engine="in-memory")`; joins `maintain_order="left"`; re-sort on explicit keys; `join_asof(by=)` with `check_sortedness=False`. DuckDB: `SET TimeZone='UTC'`, integers only.
- **Hashes**: `manifest.frame_content_sha256` (logical), never IPC bytes. **Decision arrays**: `costs.DECISION_DTYPE` ("<U7"), else numpy truncates "approve".
- **Model artifacts**: `ModelBundle` (no pickles); MLflow pyfunc wraps it. Locate the bundle with `find_bundle_dir` (MLflow stores artifacts under the source dir name). Use `mlflow_utils.client/download_model_version` (pins tracking **and** registry URI; never rely on MLflow globals). Aliases only (`@champion`/`@challenger`); promotions append to the `champion_history` model tag.
- **Promotion gate** (all must pass): PR-AUC >= champion - 0.005; net savings >= champion; single-row p95 <= 5 ms; feature-set hash matches serving + online/offline parity < 1e-6. Shadow comparison is reported, not gating. `--force` is recorded in the report.
- **MLflow tracking**: SQLite + local artifacts locally; against a server leave `mlflow_artifact_location` empty (server-side `mlflow-artifacts:/`).
- **Dependency groups**: core = data + serving (incl. `mlflow-skinny`); `train` = sklearn/optuna/mlflow/pandas/shap/matplotlib; `orchestration` = prefect. Serving must not import train/orchestration libs at module level. API image: `--no-default-groups`; trainer image adds `train` + `orchestration`.
- **Tests**: markers unit/parity/smoke/integration/mlops/monitoring/ui/slow; strict pytest, `filterwarnings=error` (only narrow, commented third-party ignores). Hypothesis profiles dev/ci; value pools overlap across columns on purpose. Integration tests run in the test container on the compose network with `--namespace` / unique model names (never flush shared state).
- **Serving**: never 5xx; feature store pinged at startup (cold connect > 10 ms budget). Benchmarks report the measured model (trees, mean leaves); synthetic data needs `label_noise` or trees are stumps.
- **Streaming**: Kafka key = uid on every topic; `transactions.raw` = exact /score body (no labels). Worker logic in plain classes; `run()` only wires Quix (state_dir under /tmp).
- **Monitoring**: dashboards are generated (`make dashboards`; a test fails if stale or if a query uses an unexported metric). Zero-initialise counter label sets that may stay 0. Drift reference windows must start after the 7-day warm-up. Drift metrics go through `reports/drift/latest.json` (Prefect runs flows in subprocesses). Check rules with `make lint-infra`.
- **Cases/labels**: every `CaseStore` passes `tests/case_contract.py`. `FRAUDSHIELD_CASES__STORE=postgres` iff the mlops profile is up (Makefile `CASE_STORE`). Analyst labels feed training sets only, never online TE (parity; ADR-0013). Bundles carry `policy_window.parquet` for the simulator.
- **UI**: npm deps exact (`.npmrc`), each version >= 14 days old when pinned; `make ui-test` (tsc + vitest) before commits touching `ui/`. The browser only calls `/api/*` (Vite proxy in dev, nginx in compose; nginx re-resolves `api` via Docker DNS). Decisions = status colours + text label.
- **Avoid `str.replace` edits with non-unique anchors** (replaced twice once: `.PHONY: mlops-demo` prefixes `mlops-demo-stack`).
- **macOS**: LightGBM needs `brew install libomp` (pytest stops with a UsageError otherwise); `make test-docker` runs in Linux. Host ports: API 8000, Kafka 19092, Redis **6380**, MLflow **5050** (AirPlay holds 5000), Postgres 5433, Prefect 4200, Console 8080, Prometheus 9090, Grafana 3000, UI 5180 (Vite dev 5173).

## Commands

```bash
make setup | lint | test | test-cov | format | test-docker        # dev loop (test-docker = Linux container)
make data | features | features-check | eda DATASET=sparkov       # Phase 1 pipeline
make tune | train | docs | retrain | promote | rollback DATASET=.. # modeling + MLOps (real data: deferred)
make mlops-demo          # ONE command: retrain -> @challenger -> gate + shadow -> promote -> rollback (synthetic, SQLite)
make bench-docker        # in-process /score latency report (synthetic production-sized bundle)
make dev-bundle | up | up-mlops | down | down-v | ps                # compose (dev bundles from the fixture)
make replay RATE=50 ARGS="--max-events 300 --namespace demo"      # SOURCE=fixture|canonical
make up-all | drift-demo | dashboards | lint-infra                 # monitoring profile; drift demo -> DriftDetected + retrain
make test-integration | test-integration-mlops | test-integration-all | mlops-demo-stack
make up-ui | ui-dev | ui-test | ui-build                          # analyst UI (compose :5180 | Vite :5173)
fraudshield canary 10    # route 10 % of uids to @challenger (needs FRAUDSHIELD_SERVING__ADMIN_TOKEN)
```

## Pinned versions (verified 2026-10-24)

| Package | Version | Notes |
|---|---|---|
| Python / uv | 3.12.15 / 0.12.23 | CI pins `UV_VERSION` |
| pydantic / pydantic-settings | 2.13.5 / 2.15.0 | capped by quixstreams 3.28.1 |
| ruff / mypy / pytest / hypothesis | 0.17.2 / 2.4.0 / 9.1.1 / 6.170.8 | mypy 2.x local_partial_types + strict_bytes; pytest native TOML + strict |
| polars / duckdb / pyarrow / numpy | 2.0.0 / 2.0.0 LTS / 25.0.1 / 2.5.4 | pyarrow capped <26 by mlflow |
| pandera / pyyaml | 0.34.1 / 6.0.3 | |
| lightgbm / scikit-learn / optuna / shap | 4.7.0 / 1.9.1 / 5.0.0 / 0.54.1 | shap: pass `rng`; optuna 5 TPE multivariate default |
| mlflow / mlflow-skinny | 3.17.2 | no file stores (>=3.13); server: `--allowed-hosts`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` |
| pandas / pandas-stubs / matplotlib | **2.3.3** / 2.3.3.260113 / 3.11.2 | pandas capped: evidently 0.7.23 imports a pandas-3-removed private API |
| evidently | 0.7.23 | `monitoring` group (worker image); import warnings filtered narrowly |
| fastapi / starlette | **0.150.0** / 1.9.0 | capped: Prefect 3.8 server breaks on FastAPI >= 0.155 (`Body(min_items=)`); native SSE |
| uvicorn / uvloop / httptools / orjson / prometheus-client | 0.54.0 / 0.23.0 / 0.9.0 / 3.13.0 / 0.26.0 | |
| quixstreams / confluent-kafka / redis / requests | 3.28.1 / 2.12.2 / 8.1.0 / 2.34.2 | confluent-kafka capped <2.13 by quix |
| prefect | 3.8.8 | `orchestration` group; 3.9 pins fastapi<0.151 too |
| dev: httpx2 / fakeredis / pre-commit | 2.16.0 / 2.40.0 / 4.7.0 | Starlette 1.9 TestClient uses httpx2 |
| psycopg[binary] / psycopg-pool | 3.3.6 / 3.3.3 | core (case store); async pool typed `AsyncConnectionPool[AsyncConnection[DictRow]]` |
| UI: node / react / vite / typescript | 24.21.0 / 19.3.0 / 8.3.4 / 7.0.2 | TS 7 native `tsc`; Vite 8 = rolldown |
| UI: react-query / react-router / recharts / vitest / jsdom | 5.104.1 / 8.4.0 / 3.10.1 / 5.0.3 / 30.1.2 | router 8: import from `react-router` (no -dom); recharts needs `react-is` |
| kagglehub / jupyterlab / ipykernel | 1.0.3 / 4.6.4 / 7.4.0 | `data` / `notebooks` groups |
| images | redpanda v26.2.2, console v3.13.1, redis 8.10.2-alpine, postgres 18.6-alpine, mlflow v3.17.2 (+psycopg2-binary 2.9.13), prefect 3.8.8-python3.12, prometheus v3.15.1, grafana 13.2.4, node 24.21.0-alpine, nginx 1.30.5-alpine | all arm64 + amd64 |
| GH Actions | checkout@v7, setup-uv@v10, setup-node@v7, upload-artifact@v7, setup-buildx@v4, build-push@v7 | SHA pinning in Phase 8 |

Evidently 0.7 API: `Report([DataDriftPreset(columns=...)]).run(current_data=Dataset, reference_data=Dataset)` -> snapshot; `snapshot.dict()['metrics']` has `DriftedColumnsCount` (share) + per-column `ValueDrift`. Feast pre-1.0 (pin exact). No Bytewax.

## Decisions (docs/adr/)

0001 stack · 0002 dependency policy · 0003 configuration · 0004 event time/entities · 0005 label delay + TE ·
0006 feature library + hashes · 0007 Polars 2/DuckDB 2 · 0008 modeling/policy/packaging · 0009 serving ·
0010 streaming + Quix caps · 0011 MLOps loop, gate, Prefect/FastAPI pin · 0012 monitoring, drift, alerts ·
[0013](docs/adr/0013-analyst-ui-and-case-store.md) analyst UI, case store, label path
