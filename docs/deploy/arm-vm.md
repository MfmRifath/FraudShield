# Deploy the full stack on an Arm VM

This guide covers the full system (Kafka, Redis, API, stream workers, MLflow, Prefect, monitoring, analyst UI) on one Linux VM, using the multi-arch images CD pushes to GHCR. It was written for an Oracle Cloud "Always Free" Ampere A1 instance. Oracle's free allowance has been 4 Arm OCPUs and 24 GB of memory in total; check Oracle's current terms. Any arm64 or amd64 VPS with at least 8 GB of RAM works the same way.

For the one-container demo instead, see [single-container.md](single-container.md).

## 1. The VM

- **Image:** Ubuntu 24.04 (aarch64).
- **Shape:** 4 OCPU / 24 GB. 2 OCPU / 12 GB is enough for `compose.lite.yml`.
- **Disk:** a 100 GB boot volume (Kafka log, Parquet decision log, MLflow artifacts).
- **Ingress (OCI security list / cloud firewall):** TCP 22 from your IP only, and TCP 80 + 443 (plus UDP 443 for HTTP/3) from anywhere. Nothing else: the compose override below publishes no other port publicly.
- **DNS:** an A record (e.g. `fraud.example.com`) pointing at the VM's public IP. Caddy needs it to get a certificate.

```bash
sudo apt-get update && sudo apt-get -y upgrade
curl -fsSL https://get.docker.com | sudo sh        # Docker Engine + compose plugin (arm64)
sudo usermod -aG docker "$USER" && newgrp docker
docker compose version                             # v2.24+ (the overrides use !reset / !override)
```

Oracle's Ubuntu images also ship iptables rules that drop inbound traffic. Allow 80 and 443:

```bash
sudo iptables -I INPUT 6 -p tcp -m multiport --dports 80,443 -j ACCEPT
sudo iptables -I INPUT 6 -p udp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 2. The repository and settings

```bash
sudo mkdir -p /opt/fraudshield && sudo chown "$USER" /opt/fraudshield
git clone https://github.com/MfmRifath/FraudShield.git /opt/fraudshield && cd /opt/fraudshield
cp infra/deploy/env.example .env && chmod 600 .env
```

Fill in `.env`:

| Variable | Value |
|---|---|
| `FS_DOMAIN` | the DNS name above |
| `FS_UI_USER`, `FS_UI_PASSWORD_HASH` | analyst login. Hash: `docker run --rm caddy:2.11.6-alpine caddy hash-password --plaintext '…'`, with every `$` doubled in `.env` |
| `FS_ADMIN_TOKEN` | `openssl rand -hex 32`: guards `/admin/*` (never exposed by Caddy) |
| `FS_PG_PASSWORD` | a real password, set **before** the first start (Postgres stores it on first init) |
| `FS_IMAGE_PREFIX`, `FS_TAG` | `ghcr.io/mfmrifath/` and `latest` (or a commit SHA) |
| `FS_CASE_STORE=postgres`, `FS_MODEL_SOURCE=mlflow` | cases in Postgres; serve `models:/fraudshield@champion` |

If the GHCR packages are private, log in once with a read-only token: `echo $TOKEN | docker login ghcr.io -u <user> --password-stdin`.

## 3. Start

```bash
C="docker compose -f compose.yml -f infra/deploy/compose.vm.yml --profile core --profile mlops --profile monitoring --profile ui"
$C pull
$C up -d --wait --no-build
$C ps
```

Before the first model is registered, `/ready` returns 503 and the API scores with rules only. To register one:

- **Real model:** follow the "Deferred real-data runs" in PLAN.md, then run `make retrain promote` against the VM's MLflow through an SSH tunnel.
- **Synthetic smoke test:** run the `mlops-demo/on-demand` deployment from the Prefect UI.

Then reload the API: `curl -X POST -H "X-Admin-Token: $FS_ADMIN_TOKEN" http://localhost:8000/admin/reload` from inside the VM (`docker compose exec api ...`), or restart `api`.

Open `https://fraud.example.com` and log in. The UI, its `/api` and the SSE live feed all go through Caddy → nginx → API.

## 4. Operator UIs over SSH

They listen on the VM's loopback only:

```bash
ssh -N -L 3000:localhost:3000 -L 4200:localhost:4200 -L 5050:localhost:5050 \
    -L 9090:localhost:9090 -L 8080:localhost:8080 user@fraud.example.com
# Grafana :3000, Prefect :4200, MLflow :5050, Prometheus :9090, Redpanda Console :8080
```

## 5. Continuous delivery

`cd.yml` pushes `ghcr.io/mfmrifath/fraudshield-{api,trainer,demo,ui,mlflow}:<sha>` on every push to `main`. To have it deploy too, set these in the GitHub repository settings:

| Kind | Name | Value |
|---|---|---|
| variable | `DEPLOY_HOST` | the VM's DNS name (enables the deploy job) |
| variable | `DEPLOY_USER` / `DEPLOY_PATH` | e.g. `ubuntu` / `/opt/fraudshield` |
| variable | `DEPLOY_COMPOSE_ARGS` | optional; default `-f compose.yml -f infra/deploy/compose.vm.yml --profile core --profile mlops --profile ui` |
| secret | `DEPLOY_SSH_KEY` | private key of a deploy key in the VM user's `authorized_keys` |
| secret | `DEPLOY_KNOWN_HOSTS` | output of `ssh-keyscan fraud.example.com` (the host key is pinned, not trusted on first use) |

The job checks out the pushed commit on the VM (so compose files match the images), then runs `pull` and `up -d --wait --no-build` with `FS_TAG=<sha>`. Rolling back is the same command with an older SHA (runbook 5.7).

For weekly retraining from GitHub (`ct.yml`), expose the Prefect API to the runner. Set `PREFECT_SERVER_API_AUTH_STRING` on the server and the `PREFECT_API_URL` / `PREFECT_API_AUTH_STRING` secrets in the repository. The worker's own schedule (Mondays 03:00 UTC) already retrains without it.

## 6. Smaller VMs

On 8–12 GB, use the lite override instead of `--profile mlops` with Postgres and the Prefect server:

```bash
docker compose -f compose.yml -f compose.lite.yml -f infra/deploy/compose.vm.yml --profile core --profile mlops --profile ui up -d --wait --no-build
```

Cases then live in memory (lost on restart), and Prefect runs without its UI.

## 7. Operations

Backups, alerts and failure handling: [runbook.md](../runbook.md). Keep `.env` out of git, rotate `FS_ADMIN_TOKEN` and the UI password periodically, and apply OS updates (`unattended-upgrades` is enabled on Ubuntu by default).
