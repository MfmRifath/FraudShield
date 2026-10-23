"""``fraudshield`` command-line entry point (wired to Makefile targets)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import get_args

from fraudshield.config import DatasetName, get_settings
from fraudshield.download import DownloadError
from fraudshield.pipeline import ReproducibilityError, build_features, fetch_raw

log = logging.getLogger("fraudshield")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraudshield")
    sub = p.add_subparsers(dest="group", required=True)

    def with_dataset(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument(
            "--dataset",
            choices=get_args(DatasetName),
            default=None,
            help="defaults to `dataset` in the active config",
        )
        return sp

    data = sub.add_parser("data", help="raw data").add_subparsers(dest="cmd", required=True)
    dl = with_dataset(data.add_parser("download", help="download + verify/record raw manifest"))
    dl.add_argument("--force", action="store_true", help="re-download even if files exist")

    feats = sub.add_parser("features", help="feature tables").add_subparsers(
        dest="cmd", required=True
    )
    build = with_dataset(feats.add_parser("build", help="raw -> canonical -> features parquet"))
    build.add_argument("--check", action="store_true", help="verify against committed manifest")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    dataset: DatasetName = args.dataset or settings.dataset
    try:
        if (args.group, args.cmd) == ("data", "download"):
            m = fetch_raw(settings, dataset, force=args.force)
            for f in m.files:
                log.info("%s: %d rows, sha256 %s", f.name, f.rows, f.sha256[:12])
        elif (args.group, args.cmd) == ("features", "build"):
            fm = build_features(settings, dataset, check=args.check)
            log.info("%s: %d rows x %d features", dataset, fm.rows, len(fm.feature_names))
    except (DownloadError, ReproducibilityError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
