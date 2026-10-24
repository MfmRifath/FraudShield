# ADR-0014: Hardening found by load testing, and the deployment shapes

- Status: Accepted
- Date: 2026-10-24
- Phase: 8

## Context

Phase 8 asks for:
- Locust scenarios (normal, no reason codes, Redis killed) with a report;
- a lite profile for 8 GB laptops;
- a single-container demo under 1 GB;
- multi-arch images, continuous delivery and training;
- a runbook and deployment guides.

The first load-test trial runs (100 req/s on the core stack, production-sized synthetic model) failed in ways no unit test had shown. Almost every request was answered on the degraded path even with Redis healthy. Those trial reports were not kept; the committed report in `reports/loadtest/` is the authority for numbers. The causes and fixes are below.

## Decisions

### Serving fixes (found by the load test)

1. **Load shape: Poisson arrivals.**
   - `constant_throughput` users spawned together fire in lockstep: 20 requests every 200 ms. That measured the load generator (client p50 40 ms against a 3 ms service time).
   - Locust's `poisson()` wait gives independent arrivals. 50 users × 2/s keeps the closed-loop effect small.
2. **Distinct counts from a last-seen index, not event lists.**
   - "Distinct cards per email domain over 7 days" read the domain's whole 7-day history on every request. A popular domain (gmail.com in IEEE-CIS; 4 domains in the synthetic data) grows by thousands of events per minute.
   - Stores now keep, per (entity, value column), *value → last time seen*: a Redis ZSET `fs:d:<entity>:<col>:<key>`.
   - A window count is one `ZCOUNT (t-w, t)`, plus one `ZSCORE` to know whether the current value is new. This is **exact**: a value occurred in `(t-w, t)` iff its last sighting before `t` did, since events are applied in order. Online/offline parity (< 1e-6, both stores) is unchanged.
   - Per-event history is now kept only for the card's amount windows. ADR-0010's planned "approximate counter" is not needed.
3. **Failure-rate circuit breaker.**
   - "5 consecutive failures, then open for 30 s" assumed independent failures. They are not: a 50 ms event-loop or VM stall times out every fetch in flight at once, so each hiccup cost 30 s of degraded scoring.
   - The breaker now opens when at least `min_calls` (10) calls in a `window_s` (2 s) window fail at a ratio of at least `failure_ratio` (0.5). It stays open for 5 s, then allows one half-open trial.
   - A dead store still opens it within about a second at 100 req/s (tested). Isolated timeout bursts never do (tested).
   - **Bug fixed:** a half-open trial cancelled mid-fetch never reported a verdict and wedged the breaker open forever. A cancelled trial now frees the slot (`abandon`). A regression test fails without the fix.
4. **Model inference off the event loop** (`serving.model_threads`, default 2).
   - The `pred_contrib` pass (about 4 ms) blocked the loop, so Redis replies that arrived in under 1 ms were processed after the 10 ms budget expired.
   - LightGBM is called through ctypes, which releases the GIL, so a two-thread pool overlaps model work with I/O. 4 threads ran 400 `pred_contrib` calls 3.6× faster than sequential, with bit-identical results.
   - A test checks that concurrent threaded scoring equals inline scoring exactly.
5. **`POST /score?explain=false`** skips model reason codes (the "no reason codes" scenario). Rule reasons are still returned.

### Load test (`make loadtest`)

- **Model:** a production-sized synthetic bundle (500 trees × 63 leaves) shaped to the Sparkov feature set, so the real stack serves it (`devtools loadtest-bundles`, label noise 5 %).
- **Payloads:** the synthetic test window, 70 % returning cards / 30 % new.
- **Where things run:**
  - Locust runs in the dev image on the compose network.
  - The orchestrator on the host kills and restarts Redis and samples `/metrics` every second.
- **Report:** two latency views (client; service-reported `latency_ms`), fallbacks and decisions from Prometheus deltas, and charts.
- **Deferred:** the real-model rerun is in PLAN.md.

### Lite profile (`compose.lite.yml`, `make up-lite`)

An override file on top of `compose.yml`:
- Redpanda `--memory=512M`.
- No Prefect server: the worker uses Prefect's ephemeral in-process API (SQLite), so the schedules still fire.
- MLflow on a SQLite file plus a local artifact volume. MLflow ≥ 3.13 has no plain file store, so this is the closest equivalent to "file store".
- No Postgres (cases in memory), no Redpanda Console, and lower memory limits.
- Compose's `!override` and `!reset` tags change profiles and dependencies.

### Demo container (`compose.demo.yml`, `make demo`, Dockerfile target `demo`)

- **Contents:** one process.
  - `fraudshield.demo` mounts the API under `/api` and serves the built UI with an SPA fallback (no path traversal).
  - It replays a window through the in-process scorer and releases labels with the streaming label emitter's rule.
- **Window:** the CC0 fixture until the real data run.
- **Model:** dev bundles trained during the image build.
- **Bounded memory:**
  - `publisher=none`: no buffer of published messages;
  - an in-memory case cap (10,000, evicted in batches);
  - history retention, and the distinct index pruned to its window.
- **Evidence:** `make demo-check` writes the memory evidence to `reports/demo/`, and CI runs the same check.
- **Hosting:** binds `$PORT` for free hosts. The admin API is disabled because no token is set.

### Delivery

- **`cd.yml`:**
  - builds five images (`api`, `trainer`, `demo`, `ui`, `mlflow`) for linux/amd64 and linux/arm64 on QEMU, and pushes them to GHCR as `<sha>` and `latest`;
  - an optional SSH deploy runs `git checkout <sha>` and `docker compose pull && up --no-build` with `FS_IMAGE_PREFIX`/`FS_TAG`;
  - the deploy runs only when `DEPLOY_HOST` is set, and the host key is pinned through the `DEPLOY_KNOWN_HOSTS` secret.
- **`ct.yml`:**
  - weekly run of the deployed Prefect `retrain-and-promote/weekly` (needs the `PREFECT_API_URL` secret);
  - a synthetic `make mlops-demo` pipeline check that always runs.
- **CI** gains the demo check. The compose integration test already ran.
- **Action pinning:** every action is pinned by commit SHA (release tag in a comment), checked with actionlint.
- **Dependabot:** covers uv, npm, Docker, compose and Actions, with `cooldown: 14 days`, which matches the ≥ 14-day rule of ADR-0013.

## Consequences

- A single-worker API on one laptop sustains the 100 req/s target with full reason codes. The latency tails come from everything sharing the machine (see the report's caveats).
- Thread-pool scoring adds a small handoff cost per request, and more threads compete for the GIL in Python-side work. Two threads were enough here; the setting is per deployment.
- The breaker no longer reacts to a single slow call, so a store that is slow, not down, is absorbed by per-request timeouts (degraded scoring for those requests) until failures reach half the window.
- The lite profile has no Prefect UI or Postgres; it is for running the system, not for operating it.
