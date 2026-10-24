# ADR-0013: Analyst UI, case store and the label path

- Status: Accepted
- Date: 2026-10-24
- Phase: 7

## Context

The spec asks for an analyst UI with five pages:
- Live Feed;
- Review Queue with F/L/E shortcuts;
- Case Detail;
- Model & Monitoring;
- Policy Simulator.

It also asks for Postgres tables `cases`, `labels` and `audit_log`, with analyst labels published to the `labels` topic. The DoD: labelling a case in the UI appears in the next `build_training_set` output.

## Decision

1. **Case store behind one protocol** (`cases/store.py`), with two implementations that run the same contract test (`tests/case_contract.py`):
   - **InMemory**: unit tests and the core-only stack.
   - **Postgres** (`cases/postgres.py`): psycopg 3 async pool.
     - The API creates the schema at startup.
     - Labelling updates `cases`, appends to `labels` and writes `audit_log` in **one transaction**.
     - Opening a case is idempotent (`ON CONFLICT DO NOTHING`), so a replay never reopens a labelled case.
     - `labels` is append-only. `build_training_set` keeps the latest label per transaction, so a relabel wins.

   `FRAUDSHIELD_CASES__STORE=postgres` is selected automatically by `make up-mlops`, `up-ui` and `up-all`: Postgres lives in the mlops profile. The core-only stack keeps the in-memory store.
2. **Training sets read analyst labels from the `labels` table** (`flows.load_analyst_labels`, via `fetch_analyst_labels`), or from the Parquet export when the store is in memory.
3. **Analyst labels do not change online target encodings.** This supersedes the last line of ADR-0010. As built, that line was broken three ways:
   - the processor crashed on analyst labels (they have no `label_available_time`);
   - the API also applied them directly, so they counted twice;
   - the chargeback for the same transaction matured later and counted again.

   More fundamentally, offline features count matured chargebacks only, and online features must match them (parity < 1e-6). Analyst labels now reach models through the training set. They are still published to the `labels` topic (the Parquet sink keeps them, for the audit trail and matured-label metrics), and the processor skips `label_source == "analyst"`.
4. **Case Detail context** is `GET /cases/{id}/context`:
   - **Timeline**: a per-card activity log (`serving/activity.py`, the last 50 events, newest first). In the stack it is a Redis list written by the processor from `decisions`, so it works with `update_store_inline=false`; in process the API writes it.
   - **Linked entities**: other cards in the feature store's device / email-domain history. This is best-effort, with a `store_available` flag.
5. **Policy Simulator on the champion's held-out window.**
   - Bundles now carry `policy_window.parquet`: amount, day, label and calibrated probability.
     - `train.py`: the test split.
     - The synthetic/dev bundle builder: the calibration split.
   - `POST /policy/simulate` reruns `decide` and `dollar_summary` with overridden costs and capacity. Nothing in production changes.
   - Older bundles without a window return 409 with a "retrain" message.
   - The window is a few columns × the test split. A real IEEE-CIS test split is about 100k rows, roughly 2 MB of Parquet.
6. **Monitoring summary from the API's own registry** (`monitoring/summary.py`), so the UI does not need Prometheus:
   - decision counts;
   - fallbacks;
   - latency quantiles interpolated from histogram buckets, as PromQL does;
   - queue depth;
   - a subset of `reports/drift/latest.json`.

   The Evidently HTML is served by basename from the report directory only (no path traversal). The API mounts `./reports` read-only.
7. **Shortcuts:** `F` fraud, `L` legit, `E` *examine* (open the case detail), `J`/`K` or the arrow keys to move. They are ignored while typing.
   - **Ambiguity:** the spec does not define E. "Escalate" would need a new case state and workflow, so E opens the case.
8. **UI stack:**
   - React 19 + Vite 8 + TypeScript 7 (native `tsc`), TanStack Query 5, Recharts 3 and React Router 8 (declarative mode, imports from `react-router`). Tests: Vitest 5 + Testing Library + jsdom.
   - Every npm version is exact (`.npmrc save-exact`, committed `package-lock.json`) and was **at least 14 days old** when pinned, as a supply-chain margin.
   - Pages other than the queue are lazy-loaded, so Recharts (about 100 kB gzip) loads only on Live Feed.
   - Decisions use the status palette (approve good, review warning, block critical) and always carry a text label.
9. **Serving the UI:**
   - nginx 1.30 (stable line) serves the static build on :5180 (`FS_UI_PORT`) and proxies `/api/` to the API.
   - SSE has buffering off and a 1 h read timeout.
   - `/api/admin/*` and `/api/metrics` return 403 on this origin.
   - The upstream is a variable, so nginx re-resolves it through Docker DNS. A recreated API container gets a new IP, and a startup-time lookup kept returning 502s.
   - For local development, `make ui-dev` runs Vite on :5173 with the same `/api` proxy.

## Consequences

- The DoD is tested twice:
  - unit: InMemory, labelled through the API;
  - integration: compose Postgres in a throwaway schema, labelled through the API, then `load_analyst_labels`, then `build_training_set`.
- The analyst id is free text remembered in the browser. There is **no authentication**; Phase 8 hardening puts the UI behind the deployment's auth proxy.
- `test-integration-mlops` used to select monitoring tests without the monitoring profile; it now excludes `monitoring` and `ui`.
