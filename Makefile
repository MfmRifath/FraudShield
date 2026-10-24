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

.PHONY: test-integration
test-integration: ## Docker compose integration tests
	$(call not_yet,Phase 4)

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

.PHONY: promote
promote: ## Run the promotion gate and move @champion
	$(call not_yet,Phase 5)

.PHONY: docs
docs: ## Render docs/model_card.md from reports/<ds>/metrics.json
	$(RUN) fraudshield model-card $(DS_ARG)

# ---------------------------------------------------------------- runtime
.PHONY: up
up: ## docker compose up (core + mlops + monitoring + ui)
	$(call not_yet,Phase 4)

.PHONY: up-lite
up-lite: ## docker compose up with the 8 GB lite profile
	$(call not_yet,Phase 8)

.PHONY: demo
demo: ## Single-container demo (FastAPI + replayer + in-memory store + UI)
	$(call not_yet,Phase 8)

.PHONY: replay
replay: ## Replay a time window into transactions.raw
	$(call not_yet,Phase 4)

.PHONY: drift-demo
drift-demo: ## Replay with injected fraud-ring drift
	$(call not_yet,Phase 6)

.PHONY: loadtest
loadtest: ## Locust scenarios + markdown report
	$(call not_yet,Phase 8)
