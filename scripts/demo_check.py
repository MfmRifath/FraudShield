"""Check the single-container demo: it serves the UI and API and stays under 1 GiB of memory.

    make demo-check   # starts the demo at an accelerated replay rate, then runs this script

Samples the container's memory (``docker stats``) and the number of scored transactions every
``--interval`` seconds for ``--minutes``, then writes ``reports/demo/demo-check.{json,md}``.
Exit status 1 if any check fails, so CI can run it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudshield.loadtest.runner import container_stats, image_size_bytes, mem_bytes

LIMIT = 2**30  # 1 GiB


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - local demo URL
        return r.status, r.read()


def _scored(base: str) -> float:
    _, body = _get(f"{base}/api/monitoring/summary")
    return float(sum(json.loads(body)["decisions"].values()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.url.rstrip("/")
    deadline = time.monotonic() + 120
    while True:
        try:
            if _get(f"{base}/health")[0] == 200:
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("demo did not become healthy within 120 s")
        time.sleep(1)
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.minutes * 60:
        stats = container_stats(args.container)
        samples.append({"t": round(time.monotonic() - t0, 1), "scored": _scored(base),
                        "mem_bytes": mem_bytes(stats["MemUsage"]), "cpu": stats.get("CPUPerc")})  # fmt: skip
        time.sleep(args.interval)
    ui_status, ui = _get(f"{base}/queue")
    checks = {
        "ui_served": ui_status == 200 and b'id="root"' in ui,
        "api_ready": _get(f"{base}/api/ready")[0] == 200,
        "cases_listed": _get(f"{base}/api/cases?limit=5")[0] == 200,
        "replaying": samples[-1]["scored"] > samples[0]["scored"],
        "peak_under_1gib": max(s["mem_bytes"] for s in samples) < LIMIT,
    }
    half = samples[len(samples) // 2 :]
    image = image_size_bytes(args.image)
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "container": args.container,
        "minutes": args.minutes,
        "rate_per_s": (samples[-1]["scored"] - samples[0]["scored"]) / max(samples[-1]["t"], 1e-9),
        "scored_total": samples[-1]["scored"],
        "peak_mem_mib": max(s["mem_bytes"] for s in samples) / 2**20,
        "first_mem_mib": samples[0]["mem_bytes"] / 2**20,
        "last_mem_mib": samples[-1]["mem_bytes"] / 2**20,
        "second_half_growth_mib": (half[-1]["mem_bytes"] - half[0]["mem_bytes"]) / 2**20,
        "image_size_mib": None if image is None else image / 2**20,
        "checks": checks,
        "samples": samples,
    }


def to_markdown(r: dict[str, Any]) -> str:
    ok = all(r["checks"].values())
    rows = "\n".join(f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in r["checks"].items())
    img = "-" if r["image_size_mib"] is None else f"{r['image_size_mib']:.0f} MiB"
    return f"""# Demo container check

Generated {r["generated_at"]} by `make demo-check` (`scripts/demo_check.py`). **{"All checks passed" if ok else "Some checks FAILED"}.**

The demo image (API + in-process replayer + in-memory stores + UI) ran for {r["minutes"]} min, replaying
{r["rate_per_s"]:.0f} transactions/s ({r["scored_total"]:,.0f} scored in total; the replayer loops over
its window with shifted times, so state keeps growing until retention bounds it).

| Memory | MiB |
|---|---:|
| first sample | {r["first_mem_mib"]:.0f} |
| peak | {r["peak_mem_mib"]:.0f} |
| last sample | {r["last_mem_mib"]:.0f} |
| growth over the second half | {r["second_half_growth_mib"]:+.0f} |

Limit: 1024 MiB (the container runs with `mem_limit: 1g`). Image size (`docker image inspect`): {img}.

| Check | Result |
|---|---|
{rows}
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8088")
    p.add_argument("--container", default="fraudshield-demo-demo-1")
    p.add_argument("--image", default="fraudshield-demo:dev")
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=Path("reports/demo"))
    args = p.parse_args()
    r = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "demo-check.json").write_text(json.dumps(r, indent=1) + "\n")
    (args.out / "demo-check.md").write_text(to_markdown(r))
    print(to_markdown(r))
    return 0 if all(r["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
