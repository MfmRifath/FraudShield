# FraudShield

**Real-time fraud detection service**: point-in-time-correct features, a calibrated LightGBM model with a cost-aware approve/review/block policy, streaming feature updates (Redpanda + Quix Streams + Redis), a champion/challenger MLOps loop (MLflow + Prefect), drift monitoring (Evidently, Prometheus, Grafana) and an analyst review UI (React).

> **Status: under construction (Phase 1 of 9 done: data and point-in-time features).** See [PLAN.md](PLAN.md) for the roadmap. No model metrics are reported yet. Every number that appears here later will be produced by a script in this repo that you can rerun.

## Quickstart (development)

Requirements: [uv](https://docs.astral.sh/uv/) ≥ 0.12.23, GNU make, git. Python 3.12 is installed by uv if missing.

```bash
make setup   # install locked dependencies + pre-commit hooks
make lint    # ruff + mypy (strict)
make test    # unit tests
make help    # all targets (later-phase targets report "not implemented yet")
```

## Data and features

| Dataset | Source | Access | Entity (`uid`) |
|---|---|---|---|
| IEEE-CIS (primary) | Kaggle competition `ieee-fraud-detection` | Kaggle token + accepted [rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) | `card1_addr1_D1n` |
| Sparkov | Kaggle dataset `kartik2112/fraud-detection` (CC0) | public, no credentials | `cc_num` |

```bash
make data DATASET=sparkov            # download + verify against data/manifests/sparkov.raw.json
make features DATASET=sparkov        # canonical + point-in-time feature parquet
make features-check DATASET=sparkov  # rebuild and verify content hashes against the committed manifest
make eda DATASET=sparkov             # run notebooks/01_eda.ipynb
```

For IEEE-CIS, set `KAGGLE_API_TOKEN` (or create `~/.kaggle/kaggle.json`) first, then run the same targets with `DATASET=ieee_cis`. Feature semantics, including windows, label delay and target encoding, are documented in [`features/definitions.py`](src/fraudshield/features/definitions.py) and ADRs 0004–0006.

## Design decisions

Architecture Decision Records live in [docs/adr/](docs/adr/README.md).

## License

MIT (pending confirmation).
