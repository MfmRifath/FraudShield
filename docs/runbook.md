# FraudShield runbook

For whoever operates the stack: how to tell whether it is healthy, what each alert means, and how to do the routine operations. Commands assume the repository root on the host. In production, replace `make ...` with the `docker compose` command each target prints (`make -n <target>`).

## 1. What runs where

| Service | Profile | Port (host) | Health | Role |
|---|---|---|---|---|
| `api` | core | 8000 | `/health` (process up), `/ready` (model loaded) | scoring, cases, SSE, monitoring summary |
| `redis` | core | 6380 | `redis-cli ping` | online feature store (ephemeral, `noeviction`) |
| `redpanda` (+ `console` :8080) | core | 19092 | `rpk cluster health` | Kafka topics `transactions.*`, `decisions`, `labels`, `shadow.scores` |
| `processor`, `gateway`, `label-emitter`, `parquet-sink`, `shadow` | core | 9101-9105 (`/metrics`) | worker `/metrics` | stream workers |
| `postgres` | mlops | 5433 | `pg_isready` | cases/labels/audit; MLflow and Prefect databases |
| `mlflow` | mlops | 5050 | `/health` | registry: `fraudshield@champion`, `@challenger` |
| `prefect-server` (UI :4200), `prefect-worker` | mlops | 4200 | `/api/health` | weekly retrain-and-promote, daily drift |
| `prometheus`, `grafana` | monitoring | 9090, 3000 | `/-/ready`, `/api/health` | metrics, 5 alert rules, 4 dashboards |
| `ui` | ui | 5180 | `/healthz` | analyst UI (nginx; `/api` proxied to the API) |

Status at a glance: `make ps`. Logs: `docker compose --profile <p> logs -f <service>`. Every service has a memory limit (`docker stats`).

## 2. Is it healthy?

1. `curl -fsS localhost:8000/ready`: 200 means a champion is loaded. 503 means no model; the API answers with rules only.
2. Grafana → **FraudShield / Service SLOs**: p95 latency, fallback rate by reason, decision mix.
3. Grafana → **Pipeline health**: consumer lag per worker; publish drops should be 0.
4. Prometheus → *Alerts*: anything firing (section 3).

`/score` never returns 5xx for internal faults. A degraded service shows up as `degraded: true` responses and `fallback_total{reason}`, not as errors.

## 3. Alerts

| Alert | Meaning | First checks | Likely causes and actions |
|---|---|---|---|
| **FallbackRateHigh** (> 1 % of requests on a fallback path for 2 m; `review_capacity` excluded) | requests scored without online features, or without a model | `fallback_total` by reason (Service SLOs dashboard) | `feature_store_error` / `circuit_open`: Redis down or unreachable → section 4.1. `feature_timeout`: Redis slow or the API saturated; check API CPU and `feature_fetch_seconds`, then scale out or lower the load. `model_error` / `no_model`: bad bundle → roll back (5.2). `internal_error`: read the API logs (stack trace per request); requests went to review |
| **ScoreLatencyP95High** (p95 > 100 ms for 5 m) | the API is slow | API CPU (`docker stats`), request rate, `feature_fetch_seconds` | Saturation: one uvicorn worker handles about 100 req/s with reason codes on a laptop (`reports/loadtest/`). Raise `FRAUDSHIELD_SERVING__MODEL_THREADS`, add API replicas, or shed `explain` |
| **ConsumerLagHigh** (> 30 s for 1 m) | a stream worker is behind | Pipeline health dashboard (which worker / topic); worker logs | Worker crash-looping (Redis or Kafka down) or too slow. Features stay correct but go stale while the processor lags. Restart the worker: `docker compose restart processor` |
| **ServiceTargetDown** (scrape failing 2 m) | the API or a worker is not answering `/metrics` | `make ps`, the container's logs | OOM-killed (check `docker inspect --format '{{.State.OOMKilled}}'`), crash loop, or a bad config after a deploy. Restart; roll back the image tag if it started with a deploy |
| **DriftDetected** | the daily drift job found feature drift > 30 % of the top-20 features, score PSI > 0.2 or a matured PR-AUC drop > 10 % | `reports/drift/latest.json`, the Evidently HTML (UI → Model & monitoring) | The job already triggered retraining. Check the retrain flow run in Prefect and the promotion gate report. Investigate the drifted columns: a data-pipeline change upstream, or a genuine fraud-pattern shift? |

Alertmanager is not part of the stack: alerts show in Prometheus and on the Grafana panels (ADR-0012).

## 4. Failure scenarios

### 4.1 Redis lost or restarted

- **Effect:** scoring continues on the degraded (request-only) model. The breaker opens within about a second and closes again about 5 s after Redis is back. `reports/loadtest/` shows the measured outage: zero failed requests.
- **State:** Redis has no persistence by design. After a restart the history windows (≤ 7 days), first-seen sets and target-encoding counts start empty, and features are biased for up to 7 days.
- **Rebuild faster:** replay the recent window, `make replay SOURCE=canonical ARGS="--start <7 days ago>"`. Note that this scores those transactions again; in production, use a dedicated backfill that writes state without scoring (not built yet; see the limitations in the README).
- **Memory full (`noeviction`):** writes fail and the API degrades instead of silently losing history. Raise `--maxmemory` and the container limit.

### 4.2 Kafka / Redpanda down

- The API keeps scoring. Published events queue in memory up to `publish_queue_size` (10,000), then drop (`publish_dropped_total`).
- Workers reconnect when Redpanda returns.
- Lost decision events mean missing history updates in Redis and missing rows in the decision log.

### 4.3 Postgres down

- The case store is unavailable: `/cases*` return errors and new review cases are not recorded.
- `/score` still answers, because case creation is a side effect after the response is computed.
- MLflow and Prefect also depend on it.
- Restore from backup (5.6).

### 4.4 A bad model was promoted

`make rollback`: `@champion` moves to the previous version in `champion_history` and the API hot-reloads. Then find out why the gate passed: each promotion writes `reports/<dataset>/promotions/<time>-v<version>.{md,json}` with every check and whether it was forced.

## 5. Routine operations

### 5.1 Retrain and promote by hand

```bash
make retrain DATASET=ieee_cis    # training set as of now -> train -> @challenger
make promote DATASET=ieee_cis    # 4-check gate vs @champion; promotes only if all pass
```

In the stack, the same runs as the Prefect deployment `retrain-and-promote/weekly` (Mondays 03:00 UTC), or on demand from the Prefect UI.

### 5.2 Roll back / canary

```bash
make rollback                     # previous champion
fraudshield canary 10             # 10 % of cards to @challenger (stable hash of the card)
fraudshield canary 0
```

Admin calls need `FRAUDSHIELD_SERVING__ADMIN_TOKEN` (header `X-Admin-Token`). `/api/admin/*` is blocked on the UI's nginx.

### 5.3 Rotate the admin token

Set a new `FS_ADMIN_TOKEN` in the deployment's `.env`, then run `docker compose up -d api`. The old token stops working when the container restarts.

### 5.4 Change cost assumptions or review capacity

1. Try the change in the UI's **Policy simulator** first: it recomputes on the champion's held-out window.
2. To apply it, edit `configs/costs.yaml` and restart the API.
3. The next retrain uses the new costs for its thresholds and gate.

### 5.5 Drift check by hand

```bash
docker compose --profile core --profile mlops exec prefect-worker \
  prefect deployment run drift/daily --watch
```

### 5.6 Back up and restore Postgres

```bash
docker compose exec -T postgres pg_dumpall -U fraudshield | gzip > backup-$(date +%F).sql.gz
gunzip -c backup-YYYY-MM-DD.sql.gz | docker compose exec -T postgres psql -U fraudshield
```

The volume `pg-data` holds cases, labels, audit, MLflow and Prefect metadata. MLflow artifacts are in the `mlflow-artifacts` volume; back both up together.

### 5.7 Deploy a new version

CD builds images on every push to `main` (`ghcr.io/<owner>/fraudshield-*:<sha>`) and, when configured, deploys over SSH. To deploy or roll back by hand:

```bash
export FS_IMAGE_PREFIX=ghcr.io/<owner>/ FS_TAG=<sha>
docker compose --profile core --profile mlops pull && docker compose --profile core --profile mlops up -d --wait --no-build
```

See [deploy/arm-vm.md](deploy/arm-vm.md).

## 6. Capacity notes

- **API:** one uvicorn worker; LightGBM runs in `model_threads` threads (default 2). The committed load test sustains 100 req/s with 3 reason codes on a shared laptop; `explain=false` roughly halves CPU per request.
- **Memory:** check with `docker stats`. Every service has a limit; `compose.lite.yml` trims the stack for an 8 GB laptop; the demo container stays under 1 GB (`reports/demo/`).
- **Redis:** per-event history is kept only for card amount windows. Distinct counts use per-entity last-seen sets, trimmed to 7 days.
