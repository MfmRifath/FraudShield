# ADR-0012: Monitoring, drift detection and alerting

- Status: Accepted
- Date: 2026-10-24
- Phase: 6

## Context

The spec asks for:
- a daily drift job (Evidently `DataDriftPreset` + score drift + decision mix) that triggers retraining when the drifted share among the top-20 features > 0.3, score PSI > 0.2, or matured-label PR-AUC drops > 10 %;
- a fraud-ring drift-injection replay;
- Prometheus scraping and alert rules;
- four provisioned Grafana dashboards.

## Decision

1. **Data source:** the served-traffic log written by the Parquet sink.
   - Decisions carry the features the champion actually saw, so drift is measured on real serving inputs and not recomputed offline.
   - The sink also logs the `labels` topic, used for the matured-label PR-AUC check.
2. **Signals** (`monitoring/drift.py`), comparing a current window to the reference window just before it, in event time:
   - **Feature drift**: Evidently 0.7 `DataDriftPreset` on the champion's top-k features by LightGBM gain. The trigger uses Evidently's `DriftedColumnsCount` share. Columns that are constant or empty in a window are excluded but still count in the denominator.
   - **Score drift**: PSI of served probabilities with 10 reference-quantile bins (ε-smoothed).
   - **Decision mix** (reported).
   - **Performance**: PR-AUC of served probabilities against labels matured by `as_of`. It is skipped, with a reason, if either window has fewer than `min_label_positives` positives.
3. **Outputs:**
   - the Evidently HTML report (git-ignored, about 5 MB);
   - `reports/drift/drift-<as_of>.json` and `latest.json`.

   Prometheus metrics come from a **file collector** in the long-lived worker process. Prefect's `serve` runs each flow run in a subprocess, so in-process gauges would be lost.
4. **Retraining:** the drift flow runs the configured flow as a subflow when any trigger fires: `retrain-and-promote` normally, `mlops-demo` (synthetic) in the compose stack until the real training run.
5. **Fraud-ring injection:** `fraudshield replay --drift SHARE`.
   - A deterministic `hash(txn_id)` share of events moves onto three shared devices and one email domain, with amount × 4 + 50, labelled fraud.
   - Ring events are namespaced like every replay.
6. **Alerts** (Prometheus rules, checked by `promtool` via `make lint-infra`):
   - p95 > 100 ms for 5 m;
   - fallback rate > 1 % for 2 m, excluding `review_capacity`, which is a policy outcome;
   - consumer lag > 30 s for 1 m;
   - API or stream-worker target down for 2 m;
   - `DriftDetected` (no `for`: the drift job already integrates over a window).

   Alertmanager is not added; alerts show in Prometheus and in Grafana panels.
7. **Dashboards are code:** `scripts/build_dashboards.py` renders the four dashboards (Service SLOs, Decision & Score mix, Champion vs Challenger, Pipeline health). A unit test fails if the committed JSON is stale or a query references a metric the services do not export.
   - Label sets for counters that may stay at zero (`publish_dropped_total`, every fallback reason) are zero-initialised, so "No data" means broken rather than "nothing happened".
   - The shadow worker now exports agreement and |Δp| metrics.
8. **Versions:**
   - **Evidently 0.7.23** in a `monitoring` group (worker image only).
   - It is incompatible with pandas 3 (it imports a removed private pandas API), so **pandas is pinned to 2.3.3**, with pandas-stubs 2.3.3.260113. Its import-time warnings (a `numpy.core` deprecation, an invalid escape in one module, litestar deprecations) get narrow pytest filters.
   - Prometheus v3.15.1, Grafana 13.2.4 (both multi-arch).

## Consequences

- **Warm-up drift is real.** First-seen flags, 7-day windows and target encodings change while a log fills up. A reference window that starts at the beginning of the log shows genuine drift: a test once flagged 36 % on "stationary" synthetic traffic for this reason. Reference windows must start after the longest history window (7 days). Tests compare post-warm-up windows, and only the current window gets the ring.
- `make drift-demo` result on the fixture with a 40 % ring:
  - feature drift 60 % of the top-20 and matured PR-AUC −21…26 % trigger; PSI 0.13–0.18 stays under 0.2;
  - the retrain flow registers a model, Prometheus fires `DriftDetected`, and all 26 dashboard panels have data.
