"""In-process latency benchmark for /score (Phase 3 DoD: p95 < 20 ms).

Model: a SYNTHETIC bundle of production-like size (default 500 trees x 63 leaves for both the
full and degraded boosters). Latency is driven by tree count/depth and per-event feature work,
not by data realism; the same script runs against a real bundle later (``--bundle``).

Paths measured on the synthetic test window (store warmed with the train+val windows first):
  * service: ScoringService.score() as timed by the service itself (no HTTP),
  * http:    POST /score through the ASGI stack in-process (TestClient wall clock),
  * degraded: feature store down -> request-only model.

Writes reports/benchmarks/serving_inprocess.{json,md}; exits 1 if service p95 >= --threshold-ms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from fraudshield.config import Settings
from fraudshield.serving.app import create_app
from fraudshield.serving.schemas import ScoreRequest
from fraudshield.synthetic import (
    ORIGIN,
    SPLITS,
    build_synthetic_bundle,
    synthetic_canonical,
    synthetic_settings,
)
from fraudshield.training.splits import assign_splits


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "txn_id",
        "uid",
        "amount",
        "product",
        "card_network",
        "card_kind",
        "p_email_domain",
        "r_email_domain",
        "device_type",
        "device_info",
        "addr",
    )
    body = {k: row.get(k) for k in keys}
    body["event_time"] = row["event_time"].isoformat()
    body["extra"] = {k.removeprefix("x_"): v for k, v in row.items() if k.startswith("x_")}
    return body


def _pct(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "n": float(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="fs-bench-"))
    settings = synthetic_settings(Settings())
    bundle = args.bundle or build_synthetic_bundle(
        settings, tmp / "bundle", rounds=args.rounds, num_leaves=args.leaves, label_noise=0.3
    )
    srv = settings.serving.model_copy(
        update={"dataset": "sparkov", "bundle_dir": Path(bundle), "load_challenger": False}
    )
    settings = settings.model_copy(update={"serving": srv})

    canon = assign_splits(synthetic_canonical(args.rows, seed=args.seed), ORIGIN, SPLITS)
    warm = canon.filter(pl.col("split").is_in(["train", "val"])).drop("day", "split").to_dicts()
    test = canon.filter(pl.col("split") == "test").drop("day", "split").head(args.n).to_dicts()

    out: dict[str, Any] = {}
    with TestClient(create_app(settings)) as client:
        svc = client.app.state.fs.service  # type: ignore[attr-defined]
        store = svc.store

        async def warm_up() -> None:
            for row in warm:
                await store.update(row)
                if row["label"] is not None:
                    attrs = {c: row.get(c) for c in store.spec.te_columns}
                    await store.add_label(
                        row["txn_id"], int(row["label"]), row["label_available_time"], attrs
                    )

        client.portal.call(warm_up)  # type: ignore[union-attr]
        reqs = [ScoreRequest(**_payload(r)) for r in test]
        for r in reqs[:50]:  # JIT-free, but warm caches / allocator
            client.portal.call(svc.score, r)  # type: ignore[union-attr]
        half = len(reqs) // 2
        service_ms = [client.portal.call(svc.score, r).latency_ms for r in reqs[50:half]]  # type: ignore[union-attr]
        http_ms = []
        for r in reqs[half:]:
            t0 = time.perf_counter()
            resp = client.post("/score", json=r.model_dump(mode="json"))
            http_ms.append((time.perf_counter() - t0) * 1e3)
            if resp.status_code != 200:
                raise RuntimeError(f"/score returned {resp.status_code}: {resp.text}")
        store.healthy = False
        degraded = [client.portal.call(svc.score, r) for r in reqs[50:half]]  # type: ignore[union-attr]
        if not all(d.degraded for d in degraded):
            raise RuntimeError("degraded benchmark did not take the degraded path")
        full = client.app.state.fs.models.champion.bundle.boosters["full"]  # type: ignore[attr-defined]
        leaves = [t["num_leaves"] for t in full.dump_model()["tree_info"]]
        out = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": {
                "source": "synthetic" if args.bundle is None else str(args.bundle),
                "trees_full": full.num_trees(),
                "mean_leaves_per_tree": float(np.mean(leaves)),
                "max_leaves": args.leaves,
                "n_features_full": full.num_feature(),
            },
            "threshold_ms": args.threshold_ms,
            "latency_ms": {
                "service": _pct(service_ms),
                "http_inprocess": _pct(http_ms),
                "service_degraded": _pct([d.latency_ms for d in degraded]),
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
                "in_container": Path("/.dockerenv").exists(),
                **{p: version(p) for p in ("lightgbm", "fastapi", "polars")},
            },
        }
    return out


def to_markdown(r: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {name} | {int(s['n'])} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {s['max']:.2f} |"
        for name, s in r["latency_ms"].items()
    )
    env = r["environment"]
    m = r["model"]
    return f"""# In-process serving latency

_Generated by `scripts/bench_serving.py` at {r["generated_at"]}. Rerun with `make bench-docker` (or `make bench` with libomp)._

Model: **{m["source"]}** bundle, {m["trees_full"]} trees (mean {m["mean_leaves_per_tree"]:.1f} leaves/tree, max {m["max_leaves"]}), {m["n_features_full"]} features.
A synthetic bundle measures the serving code path at production-like model size; it says nothing about model quality.

| path | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|---|
{rows}

- `service`: `ScoringService.score` (feature fetch, online features, rules, one booster's predict + pred_contrib reasons, policy, publish), timed inside the service.
- `http_inprocess`: `POST /score` through FastAPI/Starlette in-process (TestClient wall clock: validation + serialisation included).
- `service_degraded`: feature store down -> request-only model.

Environment: {env["platform"]} ({env["machine"]}, {env["cpu_count"]} CPUs, container={env["in_container"]}), Python {env["python"]}, lightgbm {env["lightgbm"]}, fastapi {env["fastapi"]}.
DoD threshold: service p95 < {r["threshold_ms"]} ms -> **{"PASS" if r["latency_ms"]["service"]["p95"] < r["threshold_ms"] else "FAIL"}**.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--rows", type=int, default=40_000, help="synthetic events generated")
    p.add_argument("--n", type=int, default=3_000, help="max test-window events timed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bundle", type=Path, default=None, help="benchmark an existing bundle dir instead"
    )
    p.add_argument("--threshold-ms", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=Path("reports/benchmarks"))
    args = p.parse_args()
    result = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "serving_inprocess.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "serving_inprocess.md").write_text(to_markdown(result))
    print(to_markdown(result))
    return 0 if result["latency_ms"]["service"]["p95"] < args.threshold_ms else 1


if __name__ == "__main__":
    sys.exit(main())
