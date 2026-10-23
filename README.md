# FraudShield

**Real-time fraud detection service**: point-in-time-correct features, a calibrated LightGBM model with a cost-aware approve/review/block policy, streaming feature updates (Redpanda + Quix Streams + Redis), a champion/challenger MLOps loop (MLflow + Prefect), drift monitoring (Evidently, Prometheus, Grafana) and an analyst review UI (React).

> **Status: under construction (Phase 0 of 9: foundations).** See [PLAN.md](PLAN.md) for the roadmap. No metrics are reported yet. Every number that appears here later will be produced by a script in this repo that you can rerun.

## Quickstart (development)

Requirements: [uv](https://docs.astral.sh/uv/) ≥ 0.12.23, GNU make, git. Python 3.12 is installed by uv if missing.

```bash
make setup   # install locked dependencies + pre-commit hooks
make lint    # ruff + mypy (strict)
make test    # unit tests
make help    # all targets (later-phase targets report "not implemented yet")
```

## Design decisions

Architecture Decision Records live in [docs/adr/](docs/adr/README.md).

## License

MIT (pending confirmation).
