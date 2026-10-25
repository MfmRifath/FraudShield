# ADR-0015: What the real-data runs changed

- Status: Accepted
- Date: 2026-10-25
- Phase: real-data runs (after Phase 9)

## Context

All code was written and tested on fixtures and synthetic data first (CLAUDE.md rule 9). The first real runs were:
- IEEE-CIS tuning and training (590,540 transactions, 81 features);
- Sparkov training;
- promotion;
- the load test on the champion;
- the drift job on real traffic.

They exposed five problems that synthetic data had hidden. Each one is fixed with a regression test.

## Decisions

1. **Ranking metrics use the raw model score.**
   - Isotonic calibration is a step function. On the IEEE-CIS test window it merged about 69,500 distinct scores into 117 levels.
   - The ties understated PR-AUC (0.501 → 0.484) and recall at fixed false-positive rates (request-only model at 0.1 %: 0.183 → 0.121).
   - `evaluate` and the promotion gate now compute PR-AUC, ROC-AUC, recall at FPR and card precision on the raw score, which has the same order as the probability but without ties. Brier, ECE and every decision and dollar figure use the calibrated probability.
2. **The gate's latency check times prediction, as specified.**
   - The spec gates "single-row p95 predict ≤ 5 ms". The check timed `score_one` *with* the three TreeSHAP reason codes: 24.7 ms on the tuned model (705 trees × 153 leaves), against 0.15 ms for prediction.
   - It now gates prediction, and reports the with-reasons p95 in the promotion report as information. This was changed after the first promotion attempt failed; that report is kept (`reports/ieee_cis/promotions/*-v2.md`, both runs).
3. **Explanations set serving capacity on the real model.**
   - The real-model load test (`make loadtest-ieee`, 50 req/s, the real test window) gave zero failed requests:
     - with reason codes: service p95 59 ms, about 1.4 CPU cores;
     - without (`explain=false`): service p95 5 ms.
   - Capacity planning on this model is about explanations. One API worker with two model threads serves about 50 req/s with reason codes.
   - Options if more is needed, not taken: more model threads or replicas; explaining only review/block decisions; a smaller model for explanations; a complexity cap in tuning.
4. **One MLflow registry entry per dataset.** Training Sparkov registered into the IEEE-CIS entry (`fraudshield`) and would have become its `@challenger`. The gate's feature-set check would have blocked any promotion, but aliases must never mix datasets. The Makefile now uses `fraudshield-sparkov` for Sparkov.
5. **A missing challenger bundle no longer stops the champion.** In bundle mode a configured but absent challenger directory failed the whole model load, so the API would have served rules only. It is now optional, as it already was for MLflow.
6. **The drift job's performance check uses lagged windows.**
   - With a daily job the "current" window is one day, and labels take 1–7 days to arrive, so no current label had ever matured: the PR-AUC check was **always skipped**. The Phase 6 demo used 7-day windows and so never showed it.
   - The check now compares the latest `performance_days` (7) window whose labels have matured, ending `performance_lag_days` (7) ago, with the window before it.
   - On the real IEEE-CIS test window, run daily (`make drift-real`), it catches the mid-May decline: matured PR-AUC 0.57 → 0.42, triggering on 8 of 25 days. A performance drop surfaces a week after it starts; that is the price of honest labels.
7. **Smaller choices:**
   - Sparkov is trained with the default hyperparameters, not tuned: about 3× IEEE-CIS's size, so tuning would take hours, and it is the demo model. Test PR-AUC is 0.975: Sparkov is simulated, and its fraud comes in per-card bursts that velocity features catch.
   - The real-model demo replays a 20,000-event slice of the Sparkov test window, since the whole split would not fit the 1 GB budget.
   - Tuning now logs one line per trial with an ETA; the first run was silent for 2 h 10 min.

## Consequences

- **IEEE-CIS test-window results:**
  - PR-AUC 0.501 (rules only 0.036);
  - recall 43.9 % at 1 % FPR;
  - net savings $293,050 against approve-all, under the default costs.
- **The full model does not beat the request-only one on IEEE-CIS** (0.501 vs 0.502). The dataset's anonymised C/D/V columns already encode card history, so the engineered history features carry 22 % of the gain but add little on top. On Sparkov they matter: 0.794 → 0.975.
- **The policy simulator** on the real window prices analyst capacity: 0 reviews/day gives $225,719 of savings, 200 gives $293,050, 1,000 gives $313,533.
