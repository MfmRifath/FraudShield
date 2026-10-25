# Deploy the single-container demo

The demo image runs the scoring API, an in-process replayer and the analyst UI in **one container under 1 GB of RAM** (`reports/demo/demo-check.md`). There is no Kafka, Redis or Postgres. It suits any host that runs a Docker image and gives it about 1 GB of memory.

What it does:
- It replays a window of transactions through the scorer at `FRAUDSHIELD_DEMO__RATE` per second, then loops with shifted times. The image ships the committed CC0 Sparkov sample and a model trained on it; `make demo-sparkov` mounts the trained Sparkov champion and a slice of its real test window instead.
- It releases chargeback labels with the same delay rule as the streaming stack.
- Features live in the in-memory store. Review cases are capped at 10,000, oldest out.
- The admin API is disabled (no token set) and nothing persists across restarts.

## Run it anywhere

```bash
docker run -d --name fraudshield-demo -p 8000:8000 --memory 1g ghcr.io/mfmrifath/fraudshield-demo:latest
# or from a checkout:  make demo   (builds the image; http://localhost:8088)
```

Then open `http://<host>:8000`. The API is under `/api` (for example `/api/health` and `/api/monitoring/summary`). `/health` answers for platform health checks.

Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | port to bind (most platforms set it) |
| `FRAUDSHIELD_DEMO__RATE` | `5` | replayed transactions per second |
| `FRAUDSHIELD_DEMO__LOOP` | `true` | start over when the window ends |
| `FRAUDSHIELD_DEMO__WINDOW` | fixture | path to a canonical parquet window (mount it), e.g. the Sparkov test window after the real data run |

## Platform notes

Free tiers change often; check each platform's current limits. What matters for this image: about 1 GB of memory, a long-running process (the replayer works in the background), and an HTTP port from `$PORT`.

- **Fly.io:** `fly launch --image ghcr.io/mfmrifath/fraudshield-demo:latest --internal-port 8000 --vm-memory 1024`. Keep one machine running (`min_machines_running = 1`, `auto_stop_machines = "off"`) or the replayer stops with the machine.
- **Render:** a "Web Service" from the existing image; Render sets `PORT`. Use a plan with at least 1 GB: on smaller instances the container is killed when memory runs out.
- **Google Cloud Run:** `gcloud run deploy fraudshield-demo --image ghcr.io/mfmrifath/fraudshield-demo:latest --memory 1Gi --no-cpu-throttling --min-instances 1`. Without "CPU always allocated", the background replayer only runs while a request is in flight.
- **Hugging Face Spaces (Docker):** use a `Dockerfile` with `FROM ghcr.io/mfmrifath/fraudshield-demo:latest` and `ENV PORT=7860`, and set `app_port: 7860` in the Space's README metadata.
- **A small VPS:** the `docker run` line above, behind any TLS proxy. For example, Caddy: `caddy reverse-proxy --from demo.example.com --to localhost:8000`. Caddy streams SSE without extra settings; for nginx, turn off `proxy_buffering` on `/api/events/stream` (see `ui/nginx.conf.template`).

## Check it

```bash
curl -fsS http://<host>/api/ready && curl -fsS http://<host>/api/monitoring/summary
```

The Live feed page should show decisions arriving. `make demo-check` repeats the memory measurement locally and writes `reports/demo/`.
