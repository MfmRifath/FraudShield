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
