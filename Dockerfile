# FraudShield scoring API.  Multi-arch (linux/amd64, linux/arm64): both base images are multi-arch.
#   docker build -t fraudshield-api .
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts:ro fraudshield-api
# Installs only the core dependencies (no dev/data/train/notebook groups).

FROM ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --no-editable

# ---------------------------------------------------------------- trainer (prefect worker)
FROM ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim AS build-trainer
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --group train --group orchestration --group monitoring --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --group train --group orchestration --group monitoring --no-editable

# docker build --target trainer -t fraudshield-trainer .
FROM python:3.14.7-slim-trixie AS trainer
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/reports /app/artifacts /app/mlartifacts /home/app && chown -R app:app /app /home/app
WORKDIR /app
COPY --from=build-trainer --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app tests/fixtures ./tests/fixtures
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/home/app \
    FRAUDSHIELD_CONFIG=/app/configs/base.yaml MLFLOW_DISABLE_AGENT_HINT=1
USER app
CMD ["python", "-m", "fraudshield.flows", "serve"]

# ---------------------------------------------------------------- api image (runtime = last stage)
FROM python:3.14.7-slim-trixie AS runtime-base
# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /data/stream && chown app:app /data/stream
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FRAUDSHIELD_CONFIG=/app/configs/base.yaml \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "--factory", "fraudshield.serving.app:create_app", "--host", "0.0.0.0", "--port", "8000", \
     "--loop", "uvloop", "--http", "httptools", "--workers", "1", "--no-access-log"]

# ---------------------------------------------------------------- demo (single container)
# docker build --target demo -t fraudshield-demo .   (or: make demo)
# API + in-process replayer + in-memory feature/case stores + the built UI, < 1 GB RAM.
FROM node:24.21.0-alpine AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ui/.npmrc ./
RUN npm ci --no-audit --no-fund
COPY ui/tsconfig.json ui/vite.config.ts ui/index.html ./
COPY ui/src ./src
RUN npm run build

# dev bundles trained on the committed CC0 fixture: the demo's model until the real training run
FROM build-trainer AS demo-bundle
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY configs ./configs
COPY tests/fixtures ./tests/fixtures
RUN /app/.venv/bin/python -m fraudshield.devtools dev-bundles --out /bundles

FROM runtime-base AS demo
COPY --from=ui-build --chown=app:app /ui/dist /app/ui/dist
COPY --from=demo-bundle --chown=app:app /bundles /app/artifacts/demo
COPY --chown=app:app tests/fixtures/sparkov_sample.csv /app/tests/fixtures/sparkov_sample.csv
ENV PORT=8000 \
    FRAUDSHIELD_DATASET=sparkov \
    FRAUDSHIELD_SERVING__DATASET=sparkov \
    FRAUDSHIELD_SERVING__BUNDLE_DIR=/app/artifacts/demo/bundle \
    FRAUDSHIELD_SERVING__CHALLENGER_BUNDLE_DIR=/app/artifacts/demo/challenger \
    FRAUDSHIELD_SERVING__FEATURE_STORE=memory \
    FRAUDSHIELD_SERVING__PUBLISHER=none \
    FRAUDSHIELD_SERVING__UPDATE_STORE_INLINE=true \
    FRAUDSHIELD_CASES__STORE=memory \
    FRAUDSHIELD_DEMO__UI_DIR=/app/ui/dist
# PORT: free hosts (Render, Fly.io, Cloud Run, ...) tell the container which port to bind
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["sh", "-c", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)\""]
CMD ["sh", "-c", "exec uvicorn --factory fraudshield.demo:create_demo_app --host 0.0.0.0 --port ${PORT} --loop uvloop --http httptools --no-access-log"]

# ---------------------------------------------------------------- api (default target: keep last)
FROM runtime-base AS runtime
