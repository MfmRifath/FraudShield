"""``python -m fraudshield.loadtest {run,report}`` (the stack must be up; see ``make loadtest``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fraudshield.loadtest.report import write_report
from fraudshield.loadtest.runner import RunConfig, run_all
from fraudshield.loadtest.scenarios import SCENARIOS, by_name


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="python -m fraudshield.loadtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = RunConfig()
    run = sub.add_parser("run", help="warm-up + scenarios against the running stack, then report")
    run.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS],
                     help="repeatable; default: all")  # fmt: skip
    run.add_argument("--users", type=int, default=d.users)
    run.add_argument("--rps-per-user", type=float, default=d.rps_per_user)
    run.add_argument("--duration", type=int, default=d.duration_s, help="seconds per scenario")
    run.add_argument("--warmup", type=int, default=d.warmup_s)
    run.add_argument("--run-id", default=None, help="prefix for cards/txns (default: timestamp)")
    run.add_argument("--out", type=Path, default=d.out_dir)
    run.add_argument("--metrics-url", default=d.metrics_url)
    run.add_argument("--no-charts", action="store_true")
    rep = sub.add_parser("report", help="rebuild report.md / results.json / charts from raw/")
    rep.add_argument("--out", type=Path, default=d.out_dir)
    rep.add_argument("--no-charts", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "run":
        import time

        cfg = RunConfig(
            users=args.users, rps_per_user=args.rps_per_user, duration_s=args.duration,
            warmup_s=args.warmup, run_id=args.run_id or f"lt{int(time.time())}",
            out_dir=args.out, metrics_url=args.metrics_url,
        )  # fmt: skip
        run_all(cfg, by_name(args.scenario))
    r = write_report(args.out, charts=not args.no_charts)
    for s in r["scenarios"]:
        cl = s["client"] or {}
        print(f"{s['name']:>16}: {cl.get('requests', 0):,.0f} req, p95 {cl.get('p95_ms')} ms, "
              f"failures {cl.get('failures', 0):,.0f}")  # fmt: skip
    print(f"report: {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
