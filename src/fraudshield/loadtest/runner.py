"""Run the load-test scenarios against the compose stack (``make loadtest``).

Runs on the host and needs only Docker:
* Locust runs in the dev/test image on the compose network (``http://api:8000``), so the client
  path has no Docker Desktop port forwarding in it;
* this process scrapes the API's ``/metrics`` once a second through the published port (the
  server-side view: latency histogram, decisions, fallbacks);
* for ``redis_killed`` it kills and restarts the ``redis`` service at fixed shares of the run,
  timed from the moment load actually starts.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fraudshield.loadtest.metrics import Snapshot, scrape
from fraudshield.loadtest.scenarios import Scenario

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    users: int = 50
    rps_per_user: float = 2.0  # mean Poisson rate per user -> target = users * rps_per_user
    duration_s: int = 60
    warmup_s: int = 30
    known_share: float = 0.7
    run_id: str = "lt"
    payloads: str = "artifacts/loadtest/payloads.jsonl"  # repo-relative (mounted at /app)
    bundle_dir: str = "artifacts/loadtest/bundle"
    out_dir: Path = Path("reports/loadtest")
    metrics_url: str = "http://localhost:8000/metrics"
    target_host: str = "http://api:8000"
    network: str = "fraudshield_default"
    image: str = "fraudshield-test:dev"
    compose: tuple[str, ...] = ("docker", "compose", "--profile", "core")
    api_container: str = "fraudshield-api-1"

    @property
    def target_rps(self) -> float:
        return self.users * self.rps_per_user


@dataclass
class ScenarioRun:
    scenario: Scenario
    started_at: float  # wall clock, seconds
    load_start: float | None  # first second the API counted new decisions (relative to start)
    outage: tuple[float, float] | None  # (kill, restart), relative to start
    samples: list[Snapshot] = field(default_factory=list)
    container_stats: dict[str, str] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "scenario": asdict(self.scenario),
            "started_at": self.started_at,
            "load_start": self.load_start,
            "outage": self.outage,
            "container_stats": self.container_stats,
            "samples": [
                {"t": s.t, "decisions": dict(s.decisions), "fallbacks": dict(s.fallbacks),
                 "latency_buckets": {str(k): v for k, v in s.latency_buckets.items()}}
                for s in self.samples
            ],
        }  # fmt: skip


def locust_command(cfg: RunConfig, scenario_name: str, explain: bool, run_s: int) -> list[str]:
    """``docker run`` invocation of Locust in the dev image on the compose network."""
    repo = Path.cwd()
    env = {
        "LT_PAYLOADS": cfg.payloads,
        "LT_KNOWN_SHARE": str(cfg.known_share),
        "LT_RUN_ID": cfg.run_id,
        "LT_STAGE": scenario_name,
        "LT_EXPLAIN": "true" if explain else "false",
        "LT_RPS_PER_USER": str(cfg.rps_per_user),
    }
    cmd = ["docker", "run", "--rm", "--network", cfg.network,
           "-v", f"{repo}:/app", "-v", "fraudshield-test-venv:/venv",
           "-v", "fraudshield-test-uvcache:/root/.cache/uv", "-e", "PYTHONDONTWRITEBYTECODE=1"]  # fmt: skip
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    out = cfg.out_dir.resolve()
    if not out.is_relative_to(repo.resolve()):
        raise ValueError(f"--out must be inside the repository (mounted into Locust): {out}")
    raw = (out.relative_to(repo.resolve()) / "raw" / scenario_name).as_posix()
    cmd += [cfg.image, "uv", "run", "--frozen", "--no-sync", "locust",
            "-f", "loadtest/locustfile.py", "--headless", "--only-summary",
            "--users", str(cfg.users), "--spawn-rate", str(cfg.users),
            "--run-time", f"{run_s}s", "--stop-timeout", "5", "--host", cfg.target_host,
            "--csv", f"/app/{raw}", "--csv-full-history", "--loglevel", "WARNING"]  # fmt: skip
    return cmd


class Sampler(threading.Thread):
    """Scrapes /metrics every ``interval`` seconds until stopped."""

    def __init__(self, url: str, t0: float, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.url, self.t0, self.interval = url, t0, interval
        self.samples: list[Snapshot] = []
        self.errors = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            now = time.monotonic()
            try:
                self.samples.append(scrape(self.url, now - self.t0, timeout=2.0))
            except OSError:
                self.errors += 1  # the API itself must stay up; counted and reported
            self._halt.wait(max(0.0, self.interval - (time.monotonic() - now)))

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)

    def load_started(self) -> float | None:
        """Time of the first sample with more decisions than the first one."""
        if not self.samples:
            return None
        base = self.samples[0].scored
        return next((s.t for s in self.samples if s.scored > base), None)


_UNITS = {"B": 1, "kB": 1e3, "KiB": 2**10, "MB": 1e6, "MiB": 2**20, "GB": 1e9, "GiB": 2**30}


def mem_bytes(text: str) -> float:
    """``docker stats`` memory figure (``'181.4MiB'``, or the usage half of ``'1.2GiB / 2GiB'``)."""
    used = text.split("/")[0].strip()
    for unit in sorted(_UNITS, key=len, reverse=True):
        if used.endswith(unit):
            return float(used[: -len(unit)]) * _UNITS[unit]
    raise ValueError(f"unrecognised memory figure: {text!r}")


def image_size_bytes(image: str) -> int | None:
    out = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()  # fmt: skip
    return int(out) if out.isdigit() else None


def container_stats(name: str) -> dict[str, str]:
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    try:
        d = json.loads(out.stdout.strip().splitlines()[0])
    except (IndexError, json.JSONDecodeError):
        return {}
    return {k: d[k] for k in ("MemUsage", "CPUPerc", "MemPerc") if k in d}


def run_scenario(
    cfg: RunConfig,
    scenario: Scenario,
    run_s: int,
    *,
    sh: Callable[[list[str]], None] | None = None,
) -> ScenarioRun:
    """Run one scenario; ``sh`` runs a docker command (injectable for tests)."""
    run_cmd = sh or (lambda c: subprocess.run(c, check=True))
    t0 = time.monotonic()
    sampler = Sampler(cfg.metrics_url, t0)
    sampler.start()
    loc = subprocess.Popen(locust_command(cfg, scenario.name, scenario.explain, run_s))
    outage: tuple[float, float] | None = None
    stats: dict[str, str] = {}
    try:
        load_start = _wait_for_load(sampler, loc, timeout_s=60)
        if scenario.redis_outage is not None and load_start is not None:
            kill_at = load_start + scenario.redis_outage[0] * run_s
            start_at = load_start + scenario.redis_outage[1] * run_s
            _sleep_until(t0, kill_at)
            run_cmd([*cfg.compose, "kill", "redis"])
            killed = time.monotonic() - t0
            _sleep_until(t0, start_at)
            run_cmd([*cfg.compose, "start", "redis"])
            outage = (killed, time.monotonic() - t0)
        elif load_start is not None:
            _sleep_until(t0, load_start + run_s / 2)
            stats = container_stats(cfg.api_container)
        rc = loc.wait(timeout=run_s + 120)
        if rc != 0:
            log.warning("locust exited with %s for %s", rc, scenario.name)
    finally:
        if loc.poll() is None:
            loc.kill()
        time.sleep(1.5)  # one more sample after the last requests
        sampler.stop()
    return ScenarioRun(scenario, time.time(), load_start, outage, sampler.samples, stats)


def _wait_for_load(
    sampler: Sampler, proc: subprocess.Popen[bytes], timeout_s: float
) -> float | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and proc.poll() is None:
        started = sampler.load_started()
        if started is not None:
            return started
        time.sleep(0.25)
    return sampler.load_started()


def _sleep_until(t0: float, at: float) -> None:
    time.sleep(max(0.0, at - (time.monotonic() - t0)))


def environment(cfg: RunConfig) -> dict[str, Any]:
    info = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, check=False
    )
    try:
        d = json.loads(info.stdout)
    except json.JSONDecodeError:
        d = {}
    return {
        "host_machine": platform.machine(),
        "host_system": platform.system(),
        "docker_cpus": d.get("NCPU"),
        "docker_mem_gib": round(d["MemTotal"] / 2**30, 1) if d.get("MemTotal") else None,
        "docker_arch": d.get("Architecture"),
        "docker_os": d.get("OperatingSystem"),
    }


def run_all(cfg: RunConfig, scenarios: list[Scenario]) -> dict[str, Any]:
    """Warm-up, then each scenario; writes ``raw/*.json`` next to Locust's CSVs."""
    raw = cfg.out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if cfg.warmup_s > 0:
        log.info("warm-up %d s (builds history for returning cards; not reported)", cfg.warmup_s)
        run_scenario(cfg, Scenario("warmup", "warm-up"), cfg.warmup_s)
    runs = {}
    for sc in scenarios:
        log.info("scenario %s: %d s at %.0f rps target", sc.name, cfg.duration_s, cfg.target_rps)
        r = run_scenario(cfg, sc, cfg.duration_s)
        (raw / f"{sc.name}_server.json").write_text(json.dumps(r.as_json(), indent=1) + "\n")
        runs[sc.name] = r
    meta = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "environment": environment(cfg), "scenarios": [s.name for s in scenarios]}  # fmt: skip
    (raw / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
