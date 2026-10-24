# CLAUDE.md: FraudShield project memory

> Read this file and PLAN.md before writing code in any session. Keep it under ~200 lines.

## Current phase

**Phase 4: Streaming: complete, awaiting review.** Next: Phase 5 (MLOps loop), only after the user types "continue". Real-data DoD items are deferred (rule 9).
IEEE-CIS raw data is **not downloaded yet** (needs the user's Kaggle token + accepted competition rules). Sparkov was downloaded and built in Phase 1.

## Goal

A recruiter-grade, production-style **real-time fraud detection service**: point-in-time-correct features, LightGBM with calibration and a cost-aware decision policy, streaming feature updates, a champion/challenger MLOps loop, drift monitoring and an analyst UI. It must run on a 16 GB laptop (8 GB lite profile; under 1 GB single-container demo) on arm64 and amd64.

## Working rules (from the user, non-negotiable)

1. Work **phase by phase**. At the end of each phase: run full tests + linters, update PLAN.md (progress log) and this file, write a phase summary (built / how to run / test results / known issues / decisions needing input), then **STOP until the user types "continue"**.
2. Small conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`). Never commit `data/raw`, secrets or large artifacts.
3. All business logic in `src/fraudshield`. Notebooks only import from the package.
4. Type hints everywhere; ruff + mypy `--strict` clean; pytest for everything; prefer pure functions.
5. Before adding or upgrading a dependency, check its changelog, then pin it exactly (ADR-0002).
6. For ambiguities, choose the simplest spec-consistent option, write an ADR and list it in the phase summary. Ask only when a choice is costly to reverse.
7. **Never fabricate metrics.** Every README number must come from a rerunnable script writing to `reports/`.
8. Tick PLAN.md items only when tests prove them.
9. **Code first, run later (user, 2026-10-23):** do NOT download datasets or run real training, tuning, data-driven reports or long-running jobs until *all* phases are coded. Build and verify everything against `tests/fixtures/` and synthetic data. Real-data steps (`make data`, `make features`, `make train/tune`, reports, model card from real runs) are collected in PLAN.md "Deferred real-data runs" and are done together with the user at the end. A phase's DoD items that need real data are marked *deferred*, not ticked.

## Architecture summary

```
adapters (IEEE-CIS, Sparkov) ──> canonical Transaction ──> features/definitions.py
                                                         ├─ offline.py (Polars/DuckDB, PIT) ──> training/*
                                                         └─ online.py  (FeatureStore: Redis | InMemory) ──> serving/
streaming: replayer -> transactions.raw -> API /score -> decisions -> processor (Quix) -> Redis
           label_emitter -> labels ; parquet_sink (hourly) ; shadow (@challenger -> shadow.scores)
mlops:     Prefect flows: backfill_features, build_training_set, retrain, promote, drift
registry:  MLflow, model "fraudshield", aliases @champion / @challenger (no stages)
```

- Score path: validate -> pipelined feature fetch (10 ms timeout + circuit breaker) -> hard rules -> LightGBM (num_threads=1) -> calibrator -> cost policy -> top-3 reason codes (`pred_contrib`) -> async publish.
- Fallbacks: store failure -> degraded request-only model -> rules-only. `/score` never returns 5xx for internal faults.
- Key data assumptions: IEEE-CIS event_time = 2017-12-01 + TransactionDT seconds. uid = `card1_addr1_D1n`, where D1n = floor(TransactionDT/86400) - D1. Sparkov uid = cc_num.
- Splits (IEEE-CIS days): train 1-120, val 128-150, test 158-183 (7-day label-delay gaps).

## Repository layout

```
src/fraudshield/{adapters,features,streaming,training,serving,monitoring,flows,cases}/
src/fraudshield/config.py      # Settings (pydantic-settings + YAML): datasets, splits, labels, features
src/fraudshield/schema.py      # canonical Transaction (pydantic) + canonical frame + pandera schema
src/fraudshield/labels.py      # deterministic label_available_time (shared with label emitter)
src/fraudshield/manifest.py    # raw / features manifests, logical content hash
src/fraudshield/pipeline.py    # fetch_raw, build_features (+ --check); cli.py = `fraudshield` CLI
src/fraudshield/eda.py         # EDA summaries used by notebooks/01_eda.ipynb
src/fraudshield/rules.py       # hard rules (configs/rules.yaml): row + frame evaluation (serving + rules-only baseline)
src/fraudshield/reasons.py     # reason groups/labels + top-3 grouped reasons from pred_contrib
src/fraudshield/model_bundle.py  # deployable artifact: full.txt + degraded.txt + bundle.json (no pickles)
src/fraudshield/training/      # splits, matrix (CategoryEncoder), models, calibrate, costs (policy), evaluate, tune, package (MLflow pyfunc), model_card, train
src/fraudshield/features/online.py, store.py  # OnlineSpec + compute_online (pure); FeatureStore protocol, InMemoryFeatureStore, replay()
src/fraudshield/serving/       # app (FastAPI factory), scorer (pipeline + fallbacks), models (slots, canary, reload), breaker, events (publisher, SSE), metrics, schemas
src/fraudshield/cases/store.py # Case model, CaseStore protocol, InMemoryCaseStore (Postgres in Phase 7)
src/fraudshield/features/redis_store.py  # RedisFeatureStore (async, API) + SyncRedisFeatureWriter (processor), shared _RedisOps keys
src/fraudshield/streaming/     # replayer, gateway, processor, label_emitter, parquet_sink, shadow, common (Quix app, lag metrics); `python -m fraudshield.streaming <worker>`
src/fraudshield/devtools.py    # `dev-bundles`: fixture-trained champion/challenger -> artifacts/dev (wiring only)
src/fraudshield/training/bundle_builder.py  # train_bundle(): bundle without MLflow (synthetic/dev)
compose.yml                    # profile core: redpanda, topics-init, console, redis, api, processor, gateway, label-emitter, parquet-sink, shadow
src/fraudshield/synthetic.py   # planted-signal synthetic data + build_synthetic_bundle (tests, benchmark; never a reported model)
Dockerfile                     # API image (core deps only, non-root, libgomp1); run: uvicorn --factory fraudshield.serving.app:create_app
configs/costs.yaml, configs/rules.yaml, configs/tuned/<ds>.json (written by `make tune`)
configs/                       # base.yaml (+ costs.yaml, rules.yaml, ... in later phases)
data/{raw,interim,processed}/  # git-ignored except .gitkeep: raw/<ds>/, interim/<ds>/transactions.parquet, processed/<ds>/features.parquet
data/manifests/                # COMMITTED: <ds>.raw.json, <ds>.features.json
scripts/make_sparkov_fixture.py -> tests/fixtures/sparkov_sample.csv (2,000 raw CC0 rows, committed)
tests/oracle.py, tests/strategies.py  # brute-force feature oracle + hypothesis strategies
tests/serving_support.py       # serving test helpers; tests/parity/ = online vs offline
infra/dev/Dockerfile.test      # Linux test runner (`make test-docker`)
tests/{unit,parity,integration,fixtures}/
ui/  infra/{prometheus,grafana,nginx,deploy}/  loadtest/  notebooks/  scripts/  reports/
docs/adr/  docs/img/
```

## Conventions

- **Python 3.12**, src layout, hatchling build, `uv` for everything (`uv run --frozen ...`).
- **Config:** `fraudshield.config.get_settings()`. Precedence: init > env (`FRAUDSHIELD_*`, `__` nesting) > YAML (`FRAUDSHIELD_CONFIG`, default `configs/base.yaml`) > defaults. `extra="forbid"`, frozen. New profiles must be added to `PROFILES` in `tests/unit/test_config.py`. See ADR-0003.
- **Imports:** absolute only (`ban-relative-imports = "all"`).
- **Datetimes:** always timezone-aware UTC (ruff `DTZ` enforced).
- **Tests:** markers `unit`, `parity`, `smoke`, `integration`, `slow`. pytest runs in strict mode with `filterwarnings = error`. Hypothesis profiles are `dev` (default) and `ci` (`HYPOTHESIS_PROFILE=ci`, 300 examples, derandomized).
- **Line length:** 100. Formatter: `ruff format`.
- **Makefile targets for unbuilt phases** exit 1 with "not implemented yet (planned for Phase N)".
- **Canonical frame invariants** (enforced by `validate_canonical`): fixed columns in `CANONICAL_COLUMNS` order then `x_*` Float64 extras; `event_time` Datetime(us, UTC) strictly increasing + unique (ties broken by txn_id, +k µs; ADR-0004); `label_available_time` set iff label set and > event_time.
- **Feature semantics** are documented once, in the `features/definitions.py` docstring: windows `(t-w, t]` incl. current event; TE from daily 00:00 UTC snapshot with labels available <= snapshot (ADR-0005). Any new executor must match `tests/oracle.py`.
- **Polars 2 rules** (ADR-0007): eager or `collect(engine="in-memory")`; joins `maintain_order="left"`; re-sort on an explicit key; `join_asof(by=...)` needs `check_sortedness=False` after explicit sort (warning = test failure); `str.to_datetime` not casts. DuckDB: `SET TimeZone='UTC'`, return integers only.
- **Content hashes** use `manifest.frame_content_sha256` (logical CSV rendering), never Arrow IPC bytes.
- **Decisions** are `schema.Decision` ("approve"|"review"|"block"); numpy decision arrays MUST use `costs.DECISION_DTYPE` ("<U7"): numpy otherwise truncates "approve" -> "approv" on assignment (real bug caught by tests).
- **Model artifacts**: serving loads `ModelBundle` (LightGBM text + JSON). MLflow pyfunc wraps it; promotion uses aliases `@champion`/`@challenger`. Predict with `num_threads=1`.
- **Warning filters** (`[tool.pytest] filterwarnings`): only narrow, commented third-party entries (message + category + module where possible). Never blanket-ignore.
- **LightGBM on macOS** needs `brew install libomp`; without it pytest stops with a UsageError. `make test-docker` runs everything in Linux.
- **Dependency groups**: core `[project.dependencies]` = data + serving (incl. `mlflow-skinny`); `train` group = sklearn/optuna/mlflow/pandas/shap/matplotlib. Serving code must NOT import train-group libs at module level (sklearn import in `calibrate.fit_calibrator` is lazy). Docker image: `uv sync --no-dev --no-default-groups`.
- **Online features**: new history features must work through `OnlineSpec`/`compute_online` and pass `tests/parity` (< 1e-6). Stores keep state only.
- **Serving never 5xx** for internal faults: fallbacks are feature store -> degraded booster -> rules-only; unknown error -> `review` + `fallback=internal_error`. Validation errors are 422.
- **Benchmarks** must report the model actually measured (trees, mean leaves/tree); synthetic easy data grows stumps, so `build_synthetic_bundle(label_noise=0.3)` for latency work.
- **Avoid `# fmt: skip` inside expressions** (ruff RUF028); use it only on statements.
- **Kafka keys = uid on every topic** (per-card ordering). `transactions.raw` = exact /score body (never labels); truth goes to `transactions.truth`.
- **Stream workers**: logic in plain classes (unit-tested without Kafka); `run(settings)` only wires Quix. Quix state_dir = /tmp (image is non-root).
- **Integration tests** (`-m integration`) run in the test container on the compose network (`make test-integration`); use `--namespace` isolation, never flush shared Redis.
- **Hypothesis value pools overlap across columns** on purpose (catches stores mixing per-column state).

## Commands

```bash
make setup             # uv sync --frozen + pre-commit install
make lint              # ruff check, ruff format --check, mypy (strict)
make test              # pytest -m "not integration"
make test-cov          # with coverage
make format            # ruff --fix + format
make data DATASET=sparkov       # download (anonymous for Sparkov) + verify/record raw manifest
make features DATASET=sparkov   # canonical + features parquet + features manifest (~16 s, ~3.2 GB RSS)
make features-check DATASET=sparkov  # rebuild, fail unless hashes == committed manifest
make fixture           # regenerate the Sparkov test fixture (deterministic)
make eda DATASET=sparkov        # execute notebooks/01_eda.ipynb in place
make test-docker       # lint + tests in Linux (no local libomp needed)
make bench-docker      # in-process /score latency -> reports/benchmarks/serving_inprocess.md (synthetic bundle)
make dev-bundle        # fixture-trained dev champion/challenger -> artifacts/dev (in Docker)
make up / make down / make down-v / make ps   # compose profile core (host ports: API 8000, Kafka 19092, Redis 6380, Console 8080)
make replay RATE=50 ARGS="--max-events 300 --namespace demo"   # SOURCE=fixture (default) | canonical
make test-integration  # up + end-to-end tests on the compose network
docker build -t fraudshield-api .   # API image; mount artifacts/ and set FRAUDSHIELD_CONFIG
make tune DATASET=..   # Optuna -> configs/tuned/<ds>.json      (real data: deferred)
make train DATASET=..  # train + evaluate + MLflow + @challenger (real data: deferred)
make docs DATASET=..   # docs/model_card.md from reports/<ds>/metrics.json
uv run pre-commit run --all-files
```

## Pinned versions (verified 2026-10-23)

| Package | Version | Notes |
|---|---|---|
| Python | 3.12.15 (`>=3.12,<3.13`) | |
| uv | 0.12.23 (`required-version >=0.12.23,<0.13`) | CI pins `UV_VERSION` |
| pydantic | 2.13.5 | capped <2.14 by quixstreams 3.28.1 |
| pydantic-settings | 2.15.0 | capped <2.16 by quixstreams; built-in `YamlConfigSettingsSource` |
| pyyaml | 6.0.3 | |
| ruff | 0.17.2 | 0.17 changed default rules and the default target (we set both explicitly) |
| mypy | 2.4.0 | 2.x: `local_partial_types` + `strict_bytes` on by default; native parser default |
| pytest | 9.1.1 | native `[tool.pytest]` table + `strict = true` |
| pytest-cov | 7.1.0 | |
| hypothesis | 6.170.8 | |
| pre-commit | 4.7.0 | pre-commit-hooks v6.0.0 |
| hatchling | 1.32.4 | build backend |
| polars | 2.0.0 | streaming-by-default lazy engine; see Polars 2 rules |
| duckdb | 2.0.0 (LTS) | window `count(DISTINCT)` used for distinct counts |
| pyarrow / numpy | 25.0.1 / 2.5.4 | pyarrow capped <26 by mlflow 3.17.2 |
| pandera | 0.34.1 | `pandera.polars` backend |
| lightgbm | 4.7.0 | macOS wheel needs system libomp |
| scikit-learn | 1.9.1 | logistic baseline, isotonic/Platt fitting, metrics |
| optuna | 5.0.0 | TPE multivariate by default in 5.x; seed set explicitly |
| mlflow | 3.17.2 | file stores rejected (>=3.13): SQLite locally; aliases not stages |
| pandas / pandas-stubs | 3.1.0 / 3.0.5.260914 | pyfunc I/O only |
| shap | 0.54.1 | offline only (no numba dependency in 0.54) |
| matplotlib | 3.11.2 | report plots (Agg backend) |
| fastapi / starlette | 0.162.0 / 1.9.0 | native SSE (`fastapi.sse`); `ORJSONResponse` deprecated; lifespan only (no `on_event`) |
| uvicorn / uvloop / httptools | 0.54.0 / 0.23.0 / 0.9.0 | |
| orjson / prometheus-client | 3.13.0 / 0.26.0 | |
| mlflow-skinny | 3.17.2 | serving: registry alias lookup + artifact download |
| httpx2 (dev) | 2.16.0 | Starlette 1.9 TestClient backend (replaces httpx) |
| quixstreams | 3.28.1 | `apply/update(metadata=True)` callbacks get (value, key, ts_ms, headers) |
| confluent-kafka | 2.12.2 | capped <2.13 by quixstreams; API KafkaSink + replayer |
| redis / fakeredis (dev) | 8.1.0 / 2.40.0 | `redis.asyncio` in the API, sync client in the processor |
| requests | 2.34.2 | gateway HTTP client |
| images | redpanda v26.2.2, console v3.13.1, redis 8.10.2-alpine | all arm64 + amd64 |
| kagglehub | 1.0.3 | `data` group |
| matplotlib / jupyterlab / ipykernel | 3.11.2 / 4.6.4 / 7.4.0 | `notebooks` group |
| GH Actions | checkout@v7, setup-uv@v10, upload-artifact@v7 | SHA pinning in Phase 8 |

Stack gotchas to respect when these are added later: Evidently 0.7+ new API (`from evidently import Report, Dataset, DataDefinition`; presets in `evidently.presets`; `report.run` returns a snapshot). MLflow uses aliases, not stages. Feast pre-1.0 (pin exact). No Bytewax; use Quix Streams. Redpanda needs an explicit `--memory`.

## Decisions (docs/adr/)

- [ADR-0001](docs/adr/0001-stack-choice.md): technology stack
- [ADR-0002](docs/adr/0002-dependency-policy.md): exact pins, add deps in the phase that needs them
- [ADR-0003](docs/adr/0003-configuration.md): pydantic-settings + one YAML per profile, env overrides, forbid extras
- [ADR-0004](docs/adr/0004-event-time-and-entities.md): synthetic IEEE time, uid recipes, strict event order (µs tie-break)
- [ADR-0005](docs/adr/0005-label-delay-and-target-encoding.md): hash-based label delay (uniform 1–7 d), daily-snapshot TE
- [ADR-0006](docs/adr/0006-feature-library-design.md): declarative specs, Polars+DuckDB executor, oracle tests, manifests/hashes
- [ADR-0007](docs/adr/0007-phase1-dependency-versions.md): Polars 2 / DuckDB 2 adopted with mitigations; pandas deferred
- [ADR-0008](docs/adr/0008-modeling-policy-and-packaging.md): data use per window, baselines, calibration, cost policy + capacity, metric definitions, no-pickle bundle, reason codes
- [ADR-0009](docs/adr/0009-serving-design.md): online executor/store split, pipeline + fallbacks, capacity online, model slots/canary/reload, events, dependency split, benchmark method
- [ADR-0010](docs/adr/0010-streaming-design.md): topics/keys, gateway, truth topic + label emitter, processor consistency, Redis layout, parquet/shadow, namespace isolation, Quix dependency caps
