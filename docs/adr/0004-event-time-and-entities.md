# ADR-0004: Event time, entity ids and a strict event order

- Status: Accepted
- Date: 2026-10-23
- Phase: 1

## Context

Point-in-time features need an unambiguous answer to "which events happened before this one?". Both datasets have whole-second timestamps with ties. IEEE-CIS has no real timestamps at all, and neither dataset ships a stable card/customer id.

## Decision

1. **IEEE-CIS time:** `event_time = 2017-12-01T00:00:00Z + TransactionDT seconds`. This is the community convention: TransactionDT is a delta from an undisclosed reference. Consequence: hour-of-day and weekday are relative to a synthetic clock and may be offset from real local time.
2. **Sparkov time:** `trans_date_trans_time` is interpreted as UTC wall-clock. The `unix_time` column is ignored (it is offset by years in the published file).
3. **Entity ids:** IEEE-CIS `uid = card1 + "_" + addr1 + "_" + D1n`, with `D1n = floor(TransactionDT / 86400) - D1` (a per-card "first seen" day). Missing parts are rendered `NA`. Sparkov `uid = cc_num`.
4. **Strict total order:** adapters sort by `(event_time, txn_id)` and add *k* µs to the *k*-th event that shares an instant. Canonical `event_time` is therefore strictly increasing and unique per dataset; the canonical pandera schema enforces this. Every window is half-open, `(t - w, t]`, over that order. "Strictly before plus the current event" then has exactly one meaning in the offline executor, the online executor and the replayer, because the replayer emits the canonical (tie-broken) times.
5. **Day numbers** for splits: `day = floor((event_time - origin) / 1 day)`. Origins are configured per dataset (IEEE-CIS 2017-12-01 gives `floor(TransactionDT / 86400)`; Sparkov 2018-12-31 makes 2019-01-01 day 1).

## Alternatives considered

- Keeping ties and treating same-second events as concurrent: either the window includes the future peers (leakage) or the executors disagree on peers. Rejected.
- Ordering ties by source row number: not available once data passes through Kafka. `txn_id` is.
- A different uid recipe (e.g. adding `P_emaildomain`): higher purity but more fragmentation. The spec fixes `card1_addr1_D1n`.

## Consequences

- The tie-break shifts timestamps by at most a few µs; this has no practical effect on features.
- Adapters fail loudly if more than 10⁶ events share an instant, or if sub-second source data would collide.
