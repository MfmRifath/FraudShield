# ADR-0002: Dependency pinning and incremental adoption

- Status: Accepted
- Date: 2026-10-23
- Phase: 0

## Context

The target stack is large (LightGBM, MLflow, Prefect, Evidently, Quix Streams, FastAPI, ...). Installing everything in Phase 0 would slow CI and pin versions before we've written code against them. It would also make it easy to miss breaking changes, since nothing would exercise the APIs yet.

## Decision

1. **Exact pins** (`==`) for every direct dependency in `pyproject.toml`. `uv.lock` is committed and CI uses `uv sync --frozen`.
2. **Incremental adoption:** a dependency is added in the phase that first imports it, not before. Phase 0 runtime deps are only `pydantic`, `pydantic-settings` and `pyyaml`.
3. **Before adding or upgrading** a dependency, read its changelog for breaking changes, pin the latest patch release, and record the version in CLAUDE.md ("Pinned versions"), with any gotcha noted next to it.
4. **Optional/heavy extras** (e.g. `feast`, `onnxruntime`) go into named dependency groups so the default install stays lean.
5. **Tooling versions in CI**: uv is pinned in `ci.yml` (`UV_VERSION`) and bounded in `[tool.uv] required-version`. Pre-commit runs ruff and mypy through `uv run`, so hook versions always equal the lockfile versions.
6. **GitHub Actions** are pinned by major tag for now. Moving to commit-SHA pinning is a Phase 8 hardening task.

## Alternatives considered

- Install the full stack up front: slower CI, untested pins.
- Lower-bound pins (`>=`): not reproducible; rejected by the project rules.

## Consequences

- Each phase's first commit usually includes a `chore(deps): ...` commit.
- Renovate/Dependabot can be added in Phase 8 to propose upgrades, which still go through the changelog check.
