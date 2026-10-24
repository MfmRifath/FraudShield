# Demo video script (about 4 minutes)

A screen recording for the README (`docs/img/demo.gif` for the short loop) and a longer walkthrough. Record at 1440×900 with the browser zoomed to 110 %, the terminal font at 16 pt or larger, and a light theme. Keep the cursor still between actions.

## Before recording

```bash
docker compose -f compose.demo.yml up -d --build --wait      # demo container on :8088
# for scene 6 (optional): make up-all && make drift-demo        # full stack + Grafana
```

Let the demo run for 2–3 minutes first, so the review queue and the card timelines have content.

## Scenes

| # | Time | Screen | Do | Say |
|---|---|---|---|---|
| 1 | 0:00–0:20 | terminal | show the `docker compose -f compose.demo.yml up` line and the healthy container | "FraudShield is a real-time fraud scoring service. This is the whole demo: one container, under a gigabyte, started with one command." |
| 2 | 0:20–0:50 | UI → **Live feed** | let rows stream; point at the throughput sparkline, a `block` row and its top reason | "Each transaction gets approve, review or block in a few milliseconds, with a calibrated probability and the reason behind it. The features are point-in-time: only what was known at that moment." |
| 3 | 0:50–1:40 | **Review queue** | press `J` twice, `F` on one case, `L` on the next; show the queue shrinking | "Cases an analyst should see are sorted by expected loss: probability times what we would lose. Keyboard only: F for fraud, L for legit. Each label is written with an audit entry and becomes training data." |
| 4 | 1:40–2:30 | **Case detail** (press `E`) | scroll: reasons with value vs typical, card timeline, linked cards, audit trail | "Why was this flagged? The amount is ten times this card's typical value, and the device is shared by other cards. Here is the card's last twenty transactions, and every card that used the same device this week." |
| 5 | 2:30–3:15 | **Policy simulator** | raise the chargeback fee; set review capacity to 0; reset | "Costs are a business decision, so they are a setting, not a constant. Change them and the decisions and dollars are recomputed on the model's held-out test window. Nothing in production changes until the config does." |
| 6 | 3:15–3:45 | Grafana (optional, full stack) | Service SLOs panel, then the drift demo's `DriftDetected` | "In the full stack, Prometheus and Grafana watch latency, fallbacks and drift. When a fraud ring shifts the features, the daily drift job fires and triggers retraining, and a new model is promoted only if it passes a four-check gate." |
| 7 | 3:45–4:00 | README headline table | scroll the metrics table | "Every number here comes from a script in the repo: load test, Redis outage, demo memory. The model-quality rows fill in after the real-data run." |

## GIF cut (15–20 s, no audio)

Use scenes 2 → 3 → 4: live feed for 4 s, `F` on a queue case, `E` into the detail, scroll once. Export at 960 px wide, 12 fps, under 5 MB (for example `ffmpeg -i demo.mov -vf "fps=12,scale=960:-1:flags=lanczos" -loop 0 docs/img/demo.gif`). Then replace `demo-placeholder.svg` in the README.

## Don't

- Don't show model-quality numbers from the fixture-trained demo model: it is a wiring model, not an evaluated one.
- Don't show the admin token, `.env` files or real Kaggle data.
