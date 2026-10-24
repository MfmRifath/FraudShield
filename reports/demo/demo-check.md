# Demo container check

Generated 2026-10-24T15:07:20+00:00 by `make demo-check` (`scripts/demo_check.py`). **All checks passed.**

The demo image (API + in-process replayer + in-memory stores + UI) ran for 3.0 min, replaying
50 transactions/s (9,026 scored in total; the replayer loops over
its window with shifted times, so state keeps growing until retention bounds it).

| Memory | MiB |
|---|---:|
| first sample | 185 |
| peak | 199 |
| last sample | 190 |
| growth over the second half | +2 |

Limit: 1024 MiB (the container runs with `mem_limit: 1g`). Image size (`docker image inspect`): 289 MiB.

| Check | Result |
|---|---|
| ui_served | pass |
| api_ready | pass |
| cases_listed | pass |
| replaying | pass |
| peak_under_1gib | pass |
