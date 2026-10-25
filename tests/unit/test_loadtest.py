"""Load-test helpers: payload pool, Prometheus snapshots, report building (no Locust import)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import generate_latest

from fraudshield.config import Settings
from fraudshield.loadtest.metrics import Snapshot, parse_metrics
from fraudshield.loadtest.payloads import PayloadPool, read_payloads, write_payloads
from fraudshield.loadtest.report import (
    degraded_series,
    recovery_seconds,
    to_markdown,
    write_report,
)
from fraudshield.loadtest.runner import RunConfig, locust_command
from fraudshield.loadtest.scenarios import SCENARIOS, by_name
from fraudshield.serving.metrics import ServingMetrics
from fraudshield.streaming.replayer import load_source


@pytest.fixture(scope="module")
def bodies(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    path = tmp_path_factory.mktemp("lt") / "payloads.jsonl"
    window = load_source(Settings(), "fixture", "sparkov").head(300)
    assert write_payloads(window, path) == 300
    return read_payloads(path)


@pytest.mark.unit
def test_payload_pool_mix_and_uniqueness(bodies: list[dict[str, Any]]) -> None:
    pool = PayloadPool(bodies, known_share=0.7, seed=1, run_id="r1", worker="w0")
    out = list(pool.take(5000))
    known = [b for b in out if ":new" not in b["uid"]]
    assert abs(len(known) / len(out) - 0.7) < 0.03
    assert pool.known == len(known)
    assert len({b["txn_id"] for b in out}) == len(out)
    assert all("event_time" not in b for b in out)  # the API stamps server time
    original = {b["uid"] for b in bodies}
    assert {b["uid"].removeprefix("r1:") for b in known} <= original  # returning cards
    assert len({b["uid"] for b in out if ":new" in b["uid"]}) == len(out) - len(known)
    other = PayloadPool(bodies, run_id="r1", worker="w1").next()
    assert other["txn_id"] != out[0]["txn_id"]  # workers never collide
    assert set(bodies[0]) - {"event_time"} <= set(out[0])  # still a valid /score body


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_payload_pool_rejects_bad_share(bodies: list[dict[str, Any]], bad: float) -> None:
    with pytest.raises(ValueError, match="known_share"):
        PayloadPool(bodies, known_share=bad)


def _metrics_text(
    decisions: dict[str, int], fallbacks: dict[str, int], latencies: list[float]
) -> str:
    m = ServingMetrics()
    for d, n in decisions.items():
        m.decisions.labels(decision=d).inc(n)
    for r, n in fallbacks.items():
        m.fallbacks.labels(reason=r).inc(n)
    for x in latencies:
        m.score_latency.labels(model_version="v", degraded="false").observe(x)
    return generate_latest(m.registry).decode()


@pytest.mark.unit
def test_parse_metrics_and_deltas() -> None:
    a = parse_metrics(_metrics_text({"approve": 10}, {}, [0.002] * 10), t=0.0)
    b = parse_metrics(
        _metrics_text(
            {"approve": 90, "review": 10}, {"feature_store_error": 30}, [0.002] * 50 + [0.004] * 50
        ),
        t=10.0,
    )
    d = b.minus(a)
    assert d.t == 10.0
    assert d.scored == 90
    assert d.decisions["review"] == 10
    assert d.fallbacks["feature_store_error"] == 30
    assert d.latency_buckets[math.inf] == 90
    # delta: 40 observations at 2 ms (bucket <= 2.5 ms) + 50 at 4 ms (bucket <= 5 ms)
    q = d.latency_quantiles()
    assert q["p50"] == pytest.approx(0.0025 + 0.0025 * 5 / 50)
    assert q["p95"] == pytest.approx(0.0025 + 0.0025 * 45.5 / 50)


@pytest.mark.unit
def test_degraded_series_and_recovery() -> None:
    def snap(t: float, scored: float, fb: float, cap: float = 0) -> Snapshot:
        return Snapshot(t, {"approve": scored}, {"feature_timeout": fb, "review_capacity": cap})

    # 10 req/s; outage from t=3 to t=6 (all fallbacks), recovered from t=8
    fbs = [0, 0, 0, 10, 20, 30, 40, 45, 45, 45, 45, 45]
    samples = [snap(t, 10 * t, fb, cap=t) for t, fb in enumerate(fbs)]
    ser = degraded_series(samples)
    assert [round(p["fallback_share"], 2) for p in ser[:4]] == [0, 0, 1.0, 1.0]  # cap excluded
    assert recovery_seconds(ser, restart_at=6.0) == 2.0
    assert recovery_seconds(ser[:8], restart_at=6.0) is None


@pytest.mark.unit
def test_scenarios_and_locust_command() -> None:
    assert [s.name for s in SCENARIOS] == ["normal", "no_reason_codes", "redis_killed"]
    assert by_name(["redis_killed"])[0].redis_outage == (1 / 3, 2 / 3)
    with pytest.raises(ValueError, match="unknown scenario"):
        by_name(["nope"])
    cmd = locust_command(RunConfig(users=4, rps_per_user=2.5), "no_reason_codes", False, 30)
    assert cmd[:4] == ["docker", "run", "--rm", "--network"]
    assert "LT_EXPLAIN=false" in cmd
    assert "LT_STAGE=no_reason_codes" in cmd
    i = cmd.index("--run-time")
    assert cmd[i + 1] == "30s"
    assert cmd[cmd.index("--users") + 1] == "4"
    assert cmd[cmd.index("--host") + 1] == "http://api:8000"


LOCUST_STATS = ["Type", "Name", "Request Count", "Failure Count", "Median Response Time",
                "Average Response Time", "Min Response Time", "Max Response Time",
                "Average Content Size", "Requests/s", "Failures/s", "50%", "66%", "75%", "80%",
                "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%"]  # fmt: skip
LOCUST_HISTORY = ["Timestamp", "User Count", "Type", "Name", "Requests/s", "Failures/s", "50%",
                  "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%",
                  "Total Request Count", "Total Failure Count"]  # fmt: skip


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, header, restval="0")
        w.writeheader()
        w.writerows(rows)


@pytest.mark.unit
def test_report_from_raw_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "full.txt").write_text("tree\nTree=0\nnum_leaves=63\nTree=1\nnum_leaves=31\n")
    (bundle / "bundle.json").write_text(json.dumps({"metadata": {"model_version": "syn-500x63"},
        "full": {"booster_file": "full.txt", "feature_names": ["a", "b"]}}))  # fmt: skip
    cfg = {"users": 10, "rps_per_user": 5.0, "duration_s": 60, "warmup_s": 30,
           "known_share": 0.7, "bundle_dir": str(bundle)}  # fmt: skip
    env = {"host_machine": "arm64", "host_system": "Darwin", "docker_cpus": 8,
           "docker_mem_gib": 7.7, "docker_arch": "aarch64", "docker_os": "Docker Desktop"}  # fmt: skip
    (raw / "run.json").write_text(json.dumps({"config": cfg, "environment": env,
                                              "scenarios": ["normal", "redis_killed"]}))  # fmt: skip
    for name, outage in (("normal", None), ("redis_killed", (20.0, 40.0))):
        _write_csv(raw / f"{name}_stats.csv", LOCUST_STATS, [
            {"Type": "POST", "Name": "/score", "Request Count": 3000, "Failure Count": 0,
             "Average Response Time": 6.5, "Max Response Time": 80, "Requests/s": 49.8,
             "50%": 5, "95%": 12, "99%": 25},
            {"Type": "API", "Name": "service", "Request Count": 3000, "50%": 3, "95%": 7, "99%": 9},
            {"Type": "", "Name": "Aggregated", "Request Count": 6000},
        ])  # fmt: skip
        _write_csv(raw / f"{name}_stats_history.csv", LOCUST_HISTORY, [
            {"Timestamp": 1000 + t, "Name": "/score", "Requests/s": 50, "50%": 5, "95%": 12,
             "99%": "N/A" if t == 0 else 20} for t in range(60)
        ])  # fmt: skip
        samples = []
        for t in range(62):
            fb = sum(50 for u in range(t) if outage is not None and 20 <= u < 40)
            samples.append({"t": t, "decisions": {"approve": 50.0 * t},
                            "fallbacks": {"feature_store_error": float(fb), "review_capacity": 0.0},
                            "latency_buckets": {"0.005": 40.0 * t, "inf": 50.0 * t}})  # fmt: skip
        (raw / f"{name}_server.json").write_text(json.dumps({
            "scenario": {"name": name, "title": name.replace("_", " ")}, "load_start": 1.0,
            "outage": outage, "container_stats": {"MemUsage": "300MiB / 768MiB", "CPUPerc": "85%"},
            "samples": samples,
        }))  # fmt: skip
    r = write_report(tmp_path, charts=True)
    normal, killed = r["scenarios"]
    assert normal["client"]["p95_ms"] == 12
    assert normal["service"]["p99_ms"] == 9
    assert normal["history"][0]["p99_ms"] is None  # "N/A" parsed
    assert normal["server"]["fallback_share"] == 0
    assert killed["server"]["fallbacks"] == {"feature_store_error": 1000.0}
    assert killed["recovery_s"] is not None
    assert killed["recovery_s"] <= 2
    assert r["model"] == {"model_version": "syn-500x63", "dataset": None, "created_at": None,
                          "git_sha": None, "trees_full": 2, "mean_leaves_per_tree": 47.0,
                          "features_full": 2}  # fmt: skip
    md = (tmp_path / "report.md").read_text()
    assert "**50 req/s target**" in md
    assert "| normal | 3,000 | 49.8 | 0 | 5.0 / 12.0 / 25.0 | 3.0 / 7.0 / 9.0 | 0.0 % |" in md
    assert "Redis was killed 19 s into the load" in md
    assert md == to_markdown(json.loads((tmp_path / "results.json").read_text()))
    assert (tmp_path / "latency.png").stat().st_size > 10_000
    assert (tmp_path / "redis_killed.png").stat().st_size > 10_000


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("181.4MiB / 1GiB", 181.4 * 2**20),
        ("1.5GiB / 2GiB", 1.5 * 2**30),
        ("900kB", 900e3),
        ("12B", 12),
    ],
)
def test_mem_bytes(text: str, expected: float) -> None:
    from fraudshield.loadtest.runner import mem_bytes

    assert mem_bytes(text) == pytest.approx(expected)
    with pytest.raises(ValueError, match="unrecognised"):
        mem_bytes("lots")
