# Demo container check

Generated 2026-10-25T10:05:35+00:00 by `make demo-check` (`scripts/demo_check.py`). **All checks passed.**

The demo image (API + in-process replayer + in-memory stores + UI) ran for 3.0 min, replaying
50 transactions/s (9,032 scored in total; the replayer loops over
its window with shifted times, so state keeps growing until retention bounds it).

| Memory | MiB |
|---|---:|
| first sample | 216 |
| peak | 228 |
| last sample | 228 |
| growth over the second half | +7 |

Limit: 1024 MiB (the container runs with `mem_limit: 1g`). Image size (`docker image inspect`): 289 MiB.

| Check | Result |
|---|---|
| ui_served | pass |
| api_ready | pass |
| cases_listed | pass |
| replaying | pass |
| peak_under_1gib | pass |
