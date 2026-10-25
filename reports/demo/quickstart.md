# README quickstart check

Generated 2026-10-25T10:39:10+00:00 by `make quickstart-check` (`scripts/quickstart_check.py`) on Darwin arm64, Docker 29.7.2.
**Passed: 4.2 minutes from clone to a working demo** (goal: under 10).

The script ran the README's quickstart commands in a fresh clone, building the image
from scratch (`--no-cache`):

```bash
git clone https://github.com/MfmRifath/FraudShield.git && cd FraudShield
docker compose -f compose.demo.yml up -d --build --wait
```

| Step | Seconds |
|---|---:|
| clone | 0 |
| build the demo image (UI, Python deps, demo model) | 244 |
| start until healthy | 8 |
| **total** | **253** |

| Check | Result |
|---|---|
| ui_served | pass |
| api_ready | pass |
| api_docs | pass |
| decisions_arriving | pass |

Base images were already in the local Docker cache. On a fresh machine add their download:
`node:24.21.0-alpine` 59 MiB, `ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim` size unknown (pulled by BuildKit, not in the image store), `python:3.12.15-slim-trixie` 42 MiB (uncompressed sizes; downloads are compressed and smaller).
