"""Developer utilities.

``python -m fraudshield.devtools dev-bundles`` trains a champion and a challenger bundle on the
committed 2,000-row Sparkov fixture (CC0) into ``artifacts/dev/{bundle,challenger}``. They give
the docker-compose stack a Sparkov-compatible model before the real training run; they are
wiring artifacts, never reported models.
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m fraudshield.devtools")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("dev-bundles", help="train fixture-based champion/challenger bundles")
    b.add_argument("--out", type=Path, default=Path("artifacts/dev"))
    args = p.parse_args(argv)
    if args.cmd == "dev-bundles":
        for name, path in build_dev_bundles(get_settings(), args.out).items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
