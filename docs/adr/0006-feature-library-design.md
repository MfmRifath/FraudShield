# ADR-0006: Declarative feature specs, offline executor, and reproducibility hashes

- Status: Accepted
- Date: 2026-10-23
- Phase: 1

## Context

The spec requires one set of feature definitions with two executors (offline Polars/DuckDB, online from a feature store), parity to 1e-6, and a feature build that can be reproduced and hash-checked.

## Decision

1. **Specs, not code paths:** `features/definitions.py` declares frozen dataclasses (`Passthrough`, `Log1p`, `Cents`, `HourOfDay`, `DayOfWeek`, `SameValue`, `WindowAgg`, `DistinctCount`, `FirstSeen`, `TimeSinceLast`, `Ratio`, `TargetEncoding`). Request-only specs carry both `expr()` (Polars) and `scalar(row)` (online) implementations, tested for equality. History specs are interpreted by each executor from the documented semantics.
2. **Offline executor:**
   - Polars `rolling_{sum,mean,max}_by(..., closed="right").over(entity)` for `(t - w, t]` aggregates.
   - DuckDB `count(DISTINCT v) OVER (PARTITION BY e ORDER BY ts_us RANGE BETWEEN w-1 PRECEDING AND CURRENT ROW)` on integer microseconds for distinct counts. Polars has no rolling n-unique; DuckDB 2.0 window DISTINCT is exact and fast.
   - An as-of join of cumulative daily label tables for TE.
   - The session TimeZone is forced to UTC, and only integers come back from DuckDB.
3. **Test oracle:** `tests/oracle.py` is a naive O(n²) implementation written straight from the semantics. Hypothesis checks `offline == oracle` on generated frames (biased toward exact window-boundary offsets and timestamp ties) and on the real Sparkov fixture. It also checks prefix invariance, invariance to perturbing future events, and TE invariance to immature labels. A mutation check (switching windows to `closed="both"`, or TE to floor-of-label-day) makes these tests fail as intended.
4. **Feature set hash:** `FeatureSet.spec_hash()` is a sha256 of the JSON spec list plus TE smoothing. It is logged with models in Phase 2.
5. **Manifests** (committed in `data/manifests/`):
   - `<ds>.raw.json`: file name, size, sha256 and row count for each raw file.
   - `<ds>.features.json`: raw-manifest hash, feature-set hash, label-delay config, row count, and **logical content hashes** of the canonical and feature tables, plus library versions.
6. **Content hash:** sha256 of the schema plus a canonical CSV rendering, streamed in batches (shortest round-trip floats, `\N` nulls, quoted non-numerics). Arrow IPC bytes were rejected after testing showed they differ for equal frames, depending on chunking and buffer layout.
7. `make features-check` rebuilds and compares every field except versions. Version drift is logged as a warning, and any content difference fails.

## Consequences

- Hashes are stable within a pinned environment. Different CPU architectures or library versions *may* change float bits; the manifest records versions so a mismatch can be diagnosed.
- The online executor (Phase 3/4) must keep per-entity event history for the largest window (7d), seen-value sets for first-seen flags, and the daily TE tables.
