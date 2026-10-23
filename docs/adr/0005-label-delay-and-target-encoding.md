# ADR-0005: Label availability model and point-in-time target encoding

- Status: Accepted
- Date: 2026-10-23
- Phase: 1

## Context

The datasets say whether a transaction was fraud, but not when that became known. Target-encoded features, training-set construction and the streaming label emitter all need the same answer, or offline/online skew and label leakage follow.

## Decision

1. **Deterministic delay:** `label_available_time = event_time + delay(txn_id)`. The delay comes from `blake2b(seed:txn_id)` mapped onto the configured distribution (`labels.delay`: `fixed` or `uniform`; default uniform 1–7 days, seed 17). It is at least 1 µs, so a label is never known at its own event time.
2. **Gap consistency:** config validation rejects `labels.delay.max_days > splits.gap_days`. With 7-day split gaps, every training label has matured before validation starts.
3. **Batch target encoding:** `te_<col>` is the smoothed fraud rate from a daily snapshot taken at 00:00 UTC of the event's day. It uses only labels with `label_available_time <= snapshot`:
   `te = (pos_c + m · p0) / (n_c + m)`, with `p0 = pos_all / n_all` (null if no label is visible yet) and `m = features.target_encoding_smoothing` (default 50).
   Columns: `p_email_domain`, `device_info`, `addr`. The value is null when the column value is null.
4. The daily snapshot matches how the online path will work (Phase 3/4): a batch job materialises the TE table at midnight, serving reads it for the rest of the day, and the parity test can reproduce it exactly.
5. Labels count as available from `label_available_time` regardless of value. Legitimate labels are assumed to "mature" on the same schedule as fraud labels.

## Alternatives considered

- Instant labels (`label_available_time = event_time`): leaks the future into TE and training; rejected.
- Per-event (continuous) TE: slightly fresher, but the online store would have to update on every label, which costs parity simplicity. Rejected for now.
- Realistic 30–90-day chargeback delays: this would force much wider split gaps than the spec's 7 days. Configurable later; the validator keeps the two consistent.

## Consequences

- Property tests: flipping any label not yet available at an event's snapshot leaves that event's features unchanged (`test_target_encoding_ignores_immature_labels`).
- The Phase 4 label emitter must use `fraudshield.labels.label_delay_us`, the same function.
