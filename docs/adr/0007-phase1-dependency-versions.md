# ADR-0007: Phase 1 dependency choices (Polars 2, DuckDB 2 LTS, no pandas yet)

- Status: Accepted
- Date: 2026-10-23
- Phase: 1

## Context

At install time (2026-10-23) the latest releases were **polars 2.0.0** (2026-10-06), **duckdb 2.0.0** LTS (2026-10-21) and **pandas 3.1.0** (2026-10-22). All are new major lines.

## Decision

- **polars 2.0.0.** Breaking changes that matter to us:
  - Lazy queries default to the *streaming* engine, which does not preserve row order for joins/group_by.
  - String→datetime casts must use `str.to_datetime`.
  - `read_csv` is dispatched to `scan_csv().collect()`.

  Mitigations: our code is eager or uses `collect(engine="in-memory")`; joins use `maintain_order="left"`; every result is re-sorted on an explicit key (`_rid` / `event_time`); `join_asof` with `by` passes `check_sortedness=False` after explicit sorting (the sortedness warning would fail our warnings-as-errors test config).
- **duckdb 2.0.0** (LTS, supported to at least Oct 2027). We use only `connect`, `register`, `execute("SET TimeZone = 'UTC'")` and `sql(...).pl()`; none of the removed fetch methods. DuckDB returns timestamps in the session's local zone, so we force UTC and never read timestamps back.
- **pandas: not added.** No Phase 1 code needs it (pandera's polars backend imports without it). It arrives with the first library that requires it (shap/evidently, Phase 2/6), after a pandas 3 changelog check.
- **pandera 0.34.1** (polars backend), **pyarrow 26.0.0**, **numpy 2.5.4**.
- **kagglehub 1.0.3** in a `data` dependency group. Public datasets download without credentials (verified for Sparkov); competitions need a token and accepted rules.
- **matplotlib 3.11.2, jupyterlab 4.6.4, ipykernel 7.4.0** in a `notebooks` group (not installed in images).

## Consequences

- Be careful with copy-pasted Polars 1.x snippets (`str.strptime` casts, lazy joins without `maintain_order`).
- Revisit if pandera lags Polars 2 behaviour; the canonical schema checks are small enough to rewrite as plain Polars assertions.
