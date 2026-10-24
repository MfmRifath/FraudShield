"""Developer utilities.

``python -m fraudshield.devtools dev-bundles`` trains a champion and a challenger bundle on the
committed 2,000-row Sparkov fixture (CC0) into ``artifacts/dev/{bundle,challenger}``. They give
the docker-compose stack a Sparkov-compatible model before the real training run; they are
wiring artifacts, never reported models.

``python -m fraudshield.devtools loadtest-bundles`` trains production-sized bundles (default
500 trees x 63 leaves) on synthetic data shaped like Sparkov (same feature set, so the stack
serves them) into ``artifacts/loadtest/{bundle,challenger}``, plus ``payloads.jsonl`` (the
synthetic test window as /score bodies) for ``make loadtest``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraudshield.adapters import get_adapter
from fraudshield.config import Settings, get_settings
from fraudshield.features.offline import compute_offline
from fraudshield.pipeline import feature_set_for
from fraudshield.training.bundle_builder import train_bundle
from fraudshield.training.costs import CostConfig

FIXTURE = Path("tests/fixtures/sparkov_sample.csv")


def build_dev_bundles(
    settings: Settings, out_root: Path, fixture: Path = FIXTURE
) -> dict[str, Path]:
    fs = feature_set_for(settings, "sparkov")
    feats = compute_offline(get_adapter("sparkov", settings).load([fixture]), fs)
    costs = CostConfig.load(settings.paths.costs)
    out = {}
    for name, rounds in (("bundle", 50), ("challenger", 80)):
        out[name] = train_bundle(
            fs, feats, feats, costs,
            {"dataset": "sparkov", "created_at": "dev-fixture", "model_version": f"dev-fixture-{rounds}"},
            out_root / name, lgbm=settings.training.lgbm, rounds=rounds, num_leaves=15,
        )  # fmt: skip
    return out


def build_loadtest_artifacts(
    settings: Settings,
    out_root: Path,
    *,
    rows: int = 40_000,
    rounds: int = 500,
    leaves: int = 63,
    seed: int = 0,
    label_noise: float = 0.05,
) -> dict[str, Path]:
    import polars as pl

    from fraudshield.loadtest.payloads import write_payloads
    from fraudshield.synthetic import (
        ORIGIN,
        SPLITS,
        build_synthetic_bundle,
        synthetic_canonical,
        synthetic_settings,
    )
    from fraudshield.training.splits import assign_splits

    extras = settings.datasets.sparkov.extra_columns
    syn = synthetic_settings(settings, extras)
    canon = synthetic_canonical(rows, seed, extras)
    feats = compute_offline(canon, feature_set_for(syn, "sparkov"))
    out: dict[str, Path] = {}
    for name, s in (("bundle", seed), ("challenger", seed + 1)):
        # a little label noise keeps trees full-sized (the planted signal is otherwise learnt by
        # stumps) while the fraud rate stays near the planted 8 %, so the decision mix is plausible
        out[name] = build_synthetic_bundle(
            syn, out_root / name, features=feats, rounds=rounds, num_leaves=leaves, seed=s,
            label_noise=label_noise,
        )  # fmt: skip
    test = assign_splits(canon, ORIGIN, SPLITS).filter(pl.col("split") == "test")
    out["payloads"] = out_root / "payloads.jsonl"
    write_payloads(test.drop("day", "split"), out["payloads"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m fraudshield.devtools")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("dev-bundles", help="train fixture-based champion/challenger bundles")
    b.add_argument("--out", type=Path, default=Path("artifacts/dev"))
    lt = sub.add_parser("loadtest-bundles", help="production-sized synthetic bundles + payloads")
    lt.add_argument("--out", type=Path, default=Path("artifacts/loadtest"))
    lt.add_argument("--rows", type=int, default=40_000)
    lt.add_argument("--rounds", type=int, default=500)
    lt.add_argument("--leaves", type=int, default=63)
    lt.add_argument("--label-noise", type=float, default=0.05)
    args = p.parse_args(argv)
    if args.cmd == "dev-bundles":
        for name, path in build_dev_bundles(get_settings(), args.out).items():
            print(f"{name}: {path}")
    elif args.cmd == "loadtest-bundles":
        built = build_loadtest_artifacts(
            get_settings(), args.out, rows=args.rows, rounds=args.rounds, leaves=args.leaves,
            label_noise=args.label_noise,
        )  # fmt: skip
        for name, path in built.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
