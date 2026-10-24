# ADR-0010: Streaming topology, Redis feature store and dependency caps

- Status: Accepted
- Date: 2026-10-24
- Phase: 4

## Context

Phase 4 connects the API to Redpanda and Redis: replayed traffic must be scored, the feature store kept current by a stream processor, labels released with a realistic delay, decisions logged, and the challenger shadow-scored, all reproducibly on a laptop.

## Decision

1. **Topics** (3 partitions each, created explicitly by a one-shot `rpk` service; **every key = uid**, so each card's events stay in order end to end):

   | topic | producer | consumers |
   |---|---|---|
   | `transactions.raw` | replayer | gateway |
   | `transactions.truth` | replayer | label emitter |
   | `decisions` | API | processor, parquet sink, shadow |
   | `labels` | label emitter, API (analyst labels) | processor |
   | `shadow.scores` | shadow | (Phase 5 comparison) |

2. **Gateway worker:** the spec's flow is "replayer → transactions.raw → API /score", which needs a consumer that calls the API. The gateway plays the payment gateway: it reads `transactions.raw` per partition and calls `POST /score`, retrying with backoff on connection errors or 5xx and skipping 4xx.
3. **Truth is a separate topic.** The raw topic carries exactly the `/score` body, so labels can never reach the scorer. The label emitter holds truth records in a heap and releases each one when the replay's event-time watermark reaches its `label_available_time`. That time comes from the same deterministic `fraudshield.labels` delay used offline.
4. **Processor:** decisions update entity state and labels update the target-encoding counts. With the processor running, the API's inline store update is off (`update_store_inline=false`). Consistency is eventual: an event scored before the processor has applied that card's previous event sees slightly stale history. The integration test paces the replay at 20 events/s, and served features then equal offline features exactly.
5. **RedisFeatureStore:** state model and semantics identical to the in-memory store.
   - Layout:
     - ZSET history per entity value (trimmed to retention, with TTL);
     - last-event time;
     - first-seen SETs per (entity, column, value);
     - HASHes of label counts per visible snapshot day for target encodings.
   - One pipelined round trip per fetch.
   - A shared `_RedisOps` builds the keys for both the async API reader and the sync processor writer.
   - Redis runs `noeviction`: evicting history would silently corrupt features, while a failed write surfaces as a store error and the API degrades.
6. **Parity coverage:** the parity tests run against the Redis store via fakeredis, both all-async and API-reader + processor-writer. An in-process pipeline test runs API → decision JSON round trip → processor → next event. The integration test checks served vs offline features on the real stack.
   - Mutation checks: making target encodings visible a snapshot early is caught.
   - Mixing first-seen state across columns was *not* caught until the Hypothesis value pools were made to overlap across columns; it is now.
   - Making the Redis range query's lower bound inclusive is an equivalent mutation, because `compute_online` re-applies the exact window filter.
7. **Parquet sink:** hourly partitions by event time; flushes on size, on a background timer and on shutdown. The timer was added after the first live run showed an idle stream leaving rows in memory. Features are stored as JSON text so the schema is stable across datasets.
8. **Shadow:** scores `@challenger` on the features the champion saw (from the decision event), using the degraded booster when the champion was degraded. Rules-only decisions are skipped.
9. **Lag metric:** each worker exposes `consumer_lag_seconds{worker,topic}` (now minus the Kafka message timestamp) and `messages_processed_total{worker,topic,outcome}` on :9100.
10. **Isolation for repeated runs:** `--namespace` prefixes txn ids and entity values. The integration test uses a fresh namespace, so the persistent Redis does not need flushing. Target encodings still pool across runs and are excluded from the end-to-end parity assertion.
11. **Dev bundles:** until the real training run, the stack serves a champion and challenger trained on the committed 2,000-row CC0 fixture (`make dev-bundle`, `artifacts/dev`). They are wiring artifacts, not reported models.
12. **Host ports** are configurable (`FS_*_PORT`). Redis defaults to host port **6380**, because 6379 was taken by another project on the development machine.

## Dependency caps accepted (Quix Streams 3.28.1)

Quix Streams pins `pydantic<2.14`, `pydantic-settings<2.16` and `confluent-kafka<2.13`. We moved to **pydantic 2.13.5**, **pydantic-settings 2.15.0** and **confluent-kafka 2.12.2** rather than drop Quix (which the spec requires). All existing tests pass on these versions. Revisit when Quix relaxes its pins.

## Consequences

- `make up && make replay` runs the full topology; `make test-integration` verifies it (also in CI).
- With an IEEE-scale email domain (e.g. gmail.com), the 7-day history read for distinct-uid counts is large, so per-request fetches can exceed the 10 ms budget and fall back to the degraded model. A bucketed or approximate distinct counter is the planned mitigation; Phase 8 measures it.
- Analyst labels go to the same `labels` topic. *Superseded by [ADR-0013](0013-analyst-ui-and-case-store.md):* the processor skips them, because online target encodings count chargebacks only, as offline features do.
