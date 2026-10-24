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

IT_MARKERS ?= integration and not mlops
TEST_IMAGE_RUN = docker run --rm -v "$(CURDIR)":/app -v fraudshield-test-venv:/venv \
	-v fraudshield-test-uvcache:/root/.cache/uv -e PYTHONDONTWRITEBYTECODE=1 -e MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: test-integration
test-integration: up ## Compose integration tests (test container on the compose network)
	$(TEST_IMAGE_RUN) --network fraudshield_default \
		-e FRAUDSHIELD_STREAMING__BOOTSTRAP_SERVERS=redpanda:9092 \
		-e FRAUDSHIELD_STREAMING__REDIS_URL=redis://redis:6379/0 \
		-e FRAUDSHIELD_STREAMING__API_URL=http://api:8000 \
		-e FRAUDSHIELD_SERVING__DATASET=sparkov \
		fraudshield-test:dev sh -c 'uv sync --frozen -q && uv run --frozen pytest -m "$(IT_MARKERS)" -p no:cacheprovider $(PYTEST_ARGS)'

.PHONY: test-integration-mlops
test-integration-mlops: ## Integration tests incl. the mlops profile (MLflow server, Prefect flow run)
	$(MAKE) up-mlops
	$(MAKE) test-integration IT_MARKERS="integration" PROFILES="--profile core --profile mlops"

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
	$(RUN) fraudshield train $(DS_ARG)

.PHONY: tune
tune: ## Optuna tuning (rolling-origin CV) -> configs/tuned/<ds>.json
	$(RUN) fraudshield tune $(DS_ARG)

.PHONY: retrain
retrain: ## Training set as of now -> train -> register @challenger (real data: deferred)
	$(RUN) fraudshield retrain $(DS_ARG)

.PHONY: promote
promote: ## Gate @challenger vs @champion (PR-AUC, savings, latency, parity); move @champion if it passes
	$(RUN) fraudshield promote $(DS_ARG)

.PHONY: docs
docs: ## Render docs/model_card.md from reports/<ds>/metrics.json
	$(RUN) fraudshield model-card $(DS_ARG)

# ---------------------------------------------------------------- runtime
COMPOSE ?= docker compose
PROFILES ?= --profile core
ALL_PROFILES = --profile core --profile mlops

.PHONY: dev-bundle
dev-bundle: ## Train dev champion/challenger bundles on the CC0 fixture -> artifacts/dev (in Docker)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) fraudshield-test:dev sh -c 'uv sync --frozen -q && uv run --frozen python -m fraudshield.devtools dev-bundles'

.PHONY: up
up: ## docker compose up --wait (profile core; later phases add mlops/monitoring/ui)
	@test -f artifacts/dev/bundle/bundle.json || $(MAKE) dev-bundle
	$(COMPOSE) $(PROFILES) up -d --build --wait

.PHONY: up-mlops
up-mlops: ## up with profiles core + mlops (postgres, mlflow, prefect server + worker)
	$(MAKE) up PROFILES="--profile core --profile mlops"

.PHONY: mlops-demo
mlops-demo: ## ONE command: retrain -> @challenger -> gate + shadow -> promote -> rollback (synthetic, SQLite, in Docker)
	docker build -q -t fraudshield-test:dev -f infra/dev/Dockerfile.test infra/dev >/dev/null
	$(TEST_IMAGE_RUN) -e FRAUDSHIELD_LOG_LEVEL=WARNING fraudshield-test:dev sh -c 'uv sync --frozen -q && rm -rf artifacts/mlops-demo && uv run --frozen fraudshield mlops-demo'

.PHONY: mlops-demo-stack
mlops-demo-stack: ## Same loop as a Prefect flow run on the mlops stack (MLflow server + Postgres)
	$(COMPOSE) --profile core --profile mlops exec prefect-worker prefect deployment run mlops-demo/on-demand --watch

.PHONY: rollback
rollback: ## Move @champion back to the previous version (and hot-reload the API)
	$(RUN) fraudshield rollback

.PHONY: down
down: ## Stop the stack (keeps volumes; `make down-v` also deletes them)
	$(COMPOSE) $(ALL_PROFILES) down

.PHONY: down-v
down-v: ## Stop the stack and delete its volumes
	$(COMPOSE) $(ALL_PROFILES) down -v

.PHONY: ps
ps: ## Stack status
	$(COMPOSE) $(PROFILES) ps

.PHONY: up-lite
up-lite: ## docker compose up with the 8 GB lite profile
	$(call not_yet,Phase 8)

.PHONY: demo
demo: ## Single-container demo (FastAPI + replayer + in-memory store + UI)
	$(call not_yet,Phase 8)

RATE ?= 50
SOURCE ?= fixture
.PHONY: replay
replay: ## Replay into transactions.raw (SOURCE=fixture|canonical, RATE=events/s, ARGS=...)
	$(RUN) fraudshield replay --source $(SOURCE) --rate $(RATE) $(DS_ARG) $(ARGS)

.PHONY: drift-demo
drift-demo: ## Replay with injected fraud-ring drift
	$(call not_yet,Phase 6)

.PHONY: loadtest
loadtest: ## Locust scenarios + markdown report
	$(call not_yet,Phase 8)
