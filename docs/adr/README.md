# Architecture Decision Records

One file per decision: `NNNN-short-title.md`, using [template.md](template.md).
Status: Proposed | Accepted | Superseded by ADR-NNNN.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-stack-choice.md) | Technology stack | Accepted |
| [0002](0002-dependency-policy.md) | Dependency pinning and incremental adoption | Accepted |
| [0003](0003-configuration.md) | Configuration: pydantic-settings + YAML | Accepted |
| [0004](0004-event-time-and-entities.md) | Event time, entity ids and a strict event order | Accepted |
| [0005](0005-label-delay-and-target-encoding.md) | Label availability model and PIT target encoding | Accepted |
| [0006](0006-feature-library-design.md) | Declarative feature specs, offline executor, reproducibility hashes | Accepted |
| [0007](0007-phase1-dependency-versions.md) | Phase 1 dependency choices (Polars 2, DuckDB 2, no pandas yet) | Accepted |
| [0008](0008-modeling-policy-and-packaging.md) | Modeling, decision policy and model packaging | Accepted |
| [0009](0009-serving-design.md) | Serving design (online features, fallbacks, packaging) | Accepted |
| [0010](0010-streaming-design.md) | Streaming topology, Redis feature store and dependency caps | Accepted |
| [0011](0011-mlops-loop.md) | MLOps loop: training set, registry history, promotion gate, Prefect | Accepted |
| [0012](0012-monitoring-and-drift.md) | Monitoring, drift detection and alerting | Accepted |
| [0013](0013-analyst-ui-and-case-store.md) | Analyst UI, case store and the label path | Accepted |
