# FraudShield developer entrypoints. Run `make help` for the list.
# Targets for later phases fail loudly (exit 1) until implemented, so nothing silently "succeeds".

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --frozen
PYTEST_ARGS ?=
# Dataset for data/features/eda targets; empty = `dataset` in the active config (ieee_cis).
DATASET ?=
DS_ARG := $(if $(DATASET),--dataset $(DATASET))
# One MLflow registry entry per dataset, so @champion / @challenger never mix datasets:
# IEEE-CIS (primary) is `fraudshield`, Sparkov is `fraudshield-sparkov`.
REG := $(if $(filter sparkov,$(DATASET)),FRAUDSHIELD_TRAINING__REGISTERED_MODEL=fraudshield-sparkov)

# Local secrets (git-ignored .env, KEY=value lines). Only the Kaggle token is exported, for
# `make data`; a value in .env overrides an environment variable of the same name.
-include .env
export KAGGLE_API_TOKEN

define not_yet
	@echo "make $@: not implemented yet (planned for $(1), see PLAN.md)" >&2; exit 1
endef

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------- dev loop
.PHONY: setup
setup: ## Install locked deps and git hooks
	$(UV) sync --frozen
	$(RUN) pre-commit install

.PHONY: lint
lint: ## ruff check + ruff format --check + mypy (strict)
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

.PHONY: format
format: ## Auto-fix lint issues and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

.PHONY: test
test: ## Unit + parity + smoke tests (everything except docker integration)
	$(RUN) pytest -m "not integration" $(PYTEST_ARGS)

.PHONY: test-docker
test-docker: ## lint + tests in a Linux container (needs only Docker; no local libomp)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	docker run --rm -v "$(CURDIR)":/app -v fraudshield-test-venv:/venv \
		-v fraudshield-test-uvcache:/root/.cache/uv \
		-e PYTHONDONTWRITEBYTECODE=1 -e MYPY_CACHE_DIR=/tmp/mypy -e RUFF_CACHE_DIR=/tmp/ruff \
		-e HYPOTHESIS_PROFILE -e CI fraudshield-test:dev \
		sh -c 'uv sync --frozen -q && make lint test PYTEST_ARGS="-p no:cacheprovider $(PYTEST_ARGS)"'

.PHONY: bench
bench: ## In-process /score latency benchmark -> reports/benchmarks/ (needs libomp on macOS)
	$(RUN) python scripts/bench_serving.py

.PHONY: bench-docker
bench-docker: ## Same benchmark inside the Linux test container
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	docker run --rm -v "$(CURDIR)":/app -v fraudshield-test-venv:/venv \
		-v fraudshield-test-uvcache:/root/.cache/uv -e PYTHONDONTWRITEBYTECODE=1 fraudshield-test:dev \
		sh -c 'uv sync --frozen -q && uv run --frozen python scripts/bench_serving.py'

.PHONY: test-cov
test-cov: ## Tests with coverage report
	$(RUN) pytest -m "not integration" --cov --cov-report=term-missing $(PYTEST_ARGS)

IT_MARKERS ?= integration and not mlops and not monitoring and not ui
TEST_IMAGE_RUN = docker run --rm -v "$(CURDIR)":/app -v fraudshield-test-venv:/venv \
	-v fraudshield-test-uvcache:/root/.cache/uv -e PYTHONDONTWRITEBYTECODE=1 -e MLFLOW_DISABLE_AGENT_HINT=1

# Integration tests check streaming feature correctness (served == offline for every event), so the
# stack they bring up gets a 100 ms feature budget: a CPU-starved laptop otherwise sends the odd
# event down the degraded path. The 10 ms budget is covered by the fallback tests and `make loadtest`.
test-integration test-integration-mlops test-integration-all test-integration-lite: export FS_FEATURE_TIMEOUT_MS = 100

.PHONY: test-integration
test-integration: up test-integration-run ## Compose integration tests (test container on the compose network)

.PHONY: test-integration-run
test-integration-run: ## Integration tests against whatever stack is running (no `up`)
	$(TEST_IMAGE_RUN) --network fraudshield_default \
		-e FRAUDSHIELD_STREAMING__BOOTSTRAP_SERVERS=redpanda:9092 \
		-e FRAUDSHIELD_STREAMING__REDIS_URL=redis://redis:6379/0 \
		-e FRAUDSHIELD_STREAMING__API_URL=http://api:8000 \
		-e PREFECT_API_URL=http://prefect-server:4200/api \
		-e FRAUDSHIELD_SERVING__DATASET=sparkov \
		fraudshield-test:dev sh -c 'uv sync --frozen -q && uv run --frozen pytest -m "$(IT_MARKERS)" -p no:cacheprovider $(PYTEST_ARGS)'

.PHONY: test-integration-mlops
test-integration-mlops: ## Integration tests incl. the mlops profile (MLflow server, Prefect flow run)
	$(MAKE) up-mlops
	$(MAKE) test-integration IT_MARKERS="integration and not monitoring and not ui" PROFILES="--profile core --profile mlops"

.PHONY: test-integration-all
test-integration-all: ## Integration tests for all profiles (core + mlops + monitoring + ui)
	$(MAKE) up-all
	$(MAKE) test-integration IT_MARKERS="integration" PROFILES="$(ALL_PROFILES)"

.PHONY: clean
clean: ## Remove caches and build outputs (keeps data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

# ---------------------------------------------------------------- data & modeling
.PHONY: data
data: ## Download raw data (DATASET=ieee_cis|sparkov), verify/record data/manifests/<ds>.raw.json
	$(RUN) fraudshield data download $(DS_ARG)

.PHONY: features
features: ## Build canonical + PIT feature parquet and data/manifests/<ds>.features.json
	$(RUN) fraudshield features build $(DS_ARG)

.PHONY: features-check
features-check: ## Rebuild features and fail unless hashes match the committed manifest
	$(RUN) fraudshield features build --check $(DS_ARG)

.PHONY: fixture
fixture: ## Regenerate tests/fixtures/sparkov_sample.csv from raw Sparkov (CC0)
	$(RUN) python scripts/make_sparkov_fixture.py

.PHONY: eda
eda: ## Execute notebooks/01_eda.ipynb in place (needs `make features` for DATASET)
	FRAUDSHIELD_DATASET=$(or $(DATASET),sparkov) $(UV) run --frozen --group notebooks \
		jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

.PHONY: train
train: ## Train baselines + LightGBM, evaluate on test, log to MLflow, set @challenger
	$(REG) $(RUN) fraudshield train $(DS_ARG)

.PHONY: tune
tune: ## Optuna tuning (rolling-origin CV) -> configs/tuned/<ds>.json
	$(RUN) fraudshield tune $(DS_ARG)

.PHONY: retrain
retrain: ## Training set as of now -> train -> register @challenger (real data: deferred)
	$(REG) $(RUN) fraudshield retrain $(DS_ARG)

.PHONY: promote
promote: ## Gate @challenger vs @champion (PR-AUC, savings, latency, parity); move @champion if it passes
	$(REG) $(RUN) fraudshield promote $(DS_ARG)

.PHONY: docs
docs: ## Render docs/model_card.md from reports/<ds>/metrics.json, then the README metrics
	$(RUN) fraudshield model-card $(DS_ARG)
	$(RUN) python -m fraudshield.readme

# ---------------------------------------------------------------- runtime
COMPOSE ?= docker compose
PROFILES ?= --profile core
ALL_PROFILES = --profile core --profile mlops --profile monitoring --profile ui
# Review cases live in Postgres whenever the mlops profile (which runs it) is up.
CASE_STORE ?= $(if $(findstring mlops,$(PROFILES)),postgres,memory)

.PHONY: dev-bundle
dev-bundle: ## Train dev champion/challenger bundles on the CC0 fixture -> artifacts/dev (in Docker)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) fraudshield-test:dev sh -c 'uv sync --frozen -q && uv run --frozen python -m fraudshield.devtools dev-bundles'

.PHONY: up
up: ## docker compose up --wait (profile core)
	@test -f artifacts/dev/bundle/bundle.json || $(MAKE) dev-bundle
	FS_CASE_STORE=$(CASE_STORE) $(COMPOSE) $(PROFILES) up -d --build --wait

.PHONY: up-mlops
up-mlops: ## up with profiles core + mlops (postgres, mlflow, prefect server + worker)
	$(MAKE) up PROFILES="--profile core --profile mlops"

.PHONY: up-ui
up-ui: ## up with profiles core + mlops + ui (analyst UI on :5180, cases in Postgres)
	$(MAKE) up PROFILES="--profile core --profile mlops --profile ui"

.PHONY: up-all
up-all: ## up with every profile: core + mlops + monitoring + ui
	$(MAKE) up PROFILES="$(ALL_PROFILES)"

# ---------------------------------------------------------------- analyst UI (ui/, Node 24)
.PHONY: ui-install
ui-install: ## npm ci for the analyst UI
	cd ui && npm ci --no-audit --no-fund

.PHONY: ui-dev
ui-dev: ## Vite dev server on :5173, proxying /api to the API on :8000 (run `make up` first)
	cd ui && npm run dev

.PHONY: ui-test
ui-test: ## UI typecheck + vitest
	cd ui && npm run typecheck && npm test

.PHONY: ui-build
ui-build: ## Production build of the UI -> ui/dist
	cd ui && npm run build

.PHONY: readme
readme: ## Refresh the README headline metrics from reports/ (CI checks they are current)
	$(RUN) python -m fraudshield.readme

.PHONY: quickstart-check
quickstart-check: ## Follow the README quickstart in a fresh clone (cold build) and time it -> reports/demo/
	$(RUN) python scripts/quickstart_check.py $(QUICKSTART_ARGS)

.PHONY: dashboards
dashboards: ## Regenerate infra/grafana/dashboards/*.json from scripts/build_dashboards.py
	$(RUN) python scripts/build_dashboards.py

.PHONY: lint-infra
lint-infra: ## promtool check config + alert rules (Docker)
	docker run --rm --entrypoint promtool -v "$(CURDIR)/infra/prometheus":/etc/prometheus:ro prom/prometheus:v3.15.1 \
		check config /etc/prometheus/prometheus.yml

.PHONY: mlops-demo
mlops-demo: ## ONE command: retrain -> @challenger -> gate + shadow -> promote -> rollback (synthetic, SQLite, in Docker)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) -e FRAUDSHIELD_LOG_LEVEL=WARNING fraudshield-test:dev sh -c 'uv sync --frozen -q && rm -rf artifacts/mlops-demo && uv run --frozen fraudshield mlops-demo'

.PHONY: mlops-demo-stack
mlops-demo-stack: ## Same loop as a Prefect flow run on the mlops stack (MLflow server + Postgres)
	$(COMPOSE) --profile core --profile mlops exec prefect-worker prefect deployment run mlops-demo/on-demand --watch

.PHONY: rollback
rollback: ## Move @champion back to the previous version (and hot-reload the API)
	$(REG) $(RUN) fraudshield rollback

.PHONY: down
down: ## Stop the stack (keeps volumes; `make down-v` also deletes them)
	$(COMPOSE) $(ALL_PROFILES) down

.PHONY: down-v
down-v: ## Stop the stack and delete its volumes
	$(COMPOSE) $(ALL_PROFILES) down -v
	docker volume rm -f fraudshield_mlflow-db >/dev/null  # lite-only volume

.PHONY: ps
ps: ## Stack status
	$(COMPOSE) $(PROFILES) ps

LITE = -f compose.yml -f compose.lite.yml
.PHONY: up-lite
up-lite: ## 8 GB profile: core + MLflow (SQLite) + Prefect worker (ephemeral API); no Postgres/Prefect server
	@test -f artifacts/dev/bundle/bundle.json || $(MAKE) dev-bundle
	FS_CASE_STORE=memory $(COMPOSE) $(LITE) --profile core --profile mlops up -d --build --wait

.PHONY: test-integration-lite
test-integration-lite: ## Core integration tests against the lite stack
	$(MAKE) up-lite
	$(MAKE) test-integration-run IT_MARKERS="integration and not mlops and not monitoring and not ui"

DEMO_COMPOSE = docker compose -f compose.demo.yml
.PHONY: demo
demo: ## ONE container: API + in-process replayer + in-memory stores + UI -> http://localhost:8088
	$(DEMO_COMPOSE) up -d --build --wait
	@echo "FraudShield demo: http://localhost:$${FS_DEMO_PORT:-8088}  (API under /api; stop: make demo-down)"

.PHONY: demo-down
demo-down: ## Stop the demo container
	$(DEMO_COMPOSE) down

.PHONY: demo-check
demo-check: ## Demo at 50 tx/s for 3 min: UI/API up and memory < 1 GiB -> reports/demo/
	FS_DEMO_RATE=50 $(DEMO_COMPOSE) up -d --build --wait --force-recreate
	$(RUN) python scripts/demo_check.py $(DEMO_CHECK_ARGS)

RATE ?= 50
SOURCE ?= fixture
.PHONY: replay
replay: ## Replay into transactions.raw (SOURCE=fixture|canonical, RATE=events/s, ARGS=...)
	$(RUN) fraudshield replay --source $(SOURCE) --rate $(RATE) $(DS_ARG) $(ARGS)

.PHONY: drift-demo
drift-demo: up-all ## Normal replay, then fraud-ring drift -> drift job -> DriftDetected alert -> retrain
	$(TEST_IMAGE_RUN) --network fraudshield_default -e FRAUDSHIELD_STREAMING__BOOTSTRAP_SERVERS=redpanda:9092 \
		-e PREFECT_API_URL=http://prefect-server:4200/api fraudshield-test:dev \
		sh -c 'uv sync --frozen -q && uv run --frozen python scripts/drift_demo.py'

.PHONY: loadtest-bundles
loadtest-bundles: ## Production-sized synthetic bundles + payloads -> artifacts/loadtest (in Docker)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) fraudshield-test:dev sh -c 'uv sync --frozen -q && uv run --frozen python -m fraudshield.devtools loadtest-bundles'

LOADTEST_ARGS ?=
.PHONY: loadtest
loadtest: ## Locust: normal / no reason codes / Redis killed -> reports/loadtest (LOADTEST_ARGS="--users 20 ...")
	@test -f artifacts/loadtest/bundle/bundle.json || $(MAKE) loadtest-bundles
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) fraudshield-test:dev uv sync --frozen -q  # venv volume used by the Locust container
	FS_BUNDLE_SET=loadtest $(MAKE) up
	$(RUN) python -m fraudshield.loadtest run $(LOADTEST_ARGS)
	@echo "The stack still serves the load-test bundle; 'make up' switches back to the dev bundle."
