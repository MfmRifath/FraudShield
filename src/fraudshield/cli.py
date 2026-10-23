"""``fraudshield`` command-line entry point (wired to Makefile targets)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import get_args

from fraudshield.config import DatasetName, Settings, get_settings
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

    train = with_dataset(
        sub.add_parser("train", help="train, evaluate, log to MLflow, @challenger")
    )
    train.add_argument("--no-register", action="store_true", help="skip model registration")
    tune = with_dataset(sub.add_parser("tune", help="Optuna tuning -> configs/tuned/<ds>.json"))
    tune.add_argument("--trials", type=int, default=None, help="override training.tune.n_trials")
    with_dataset(sub.add_parser("model-card", help="render docs/model_card.md from the last run"))
    return p


def _train(settings: Settings, dataset: DatasetName, register: bool) -> None:
    from fraudshield.training.train import run_training

    r = run_training(settings, dataset, register=register)
    log.info("run %s, model version %s, report %s", r.run_id, r.model_version, r.report_dir)


def _tune(settings: Settings, dataset: DatasetName, trials: int | None) -> None:
    import polars as pl

    from fraudshield.pipeline import DatasetPaths, feature_set_for
    from fraudshield.training.train import prepare_splits
    from fraudshield.training.tune import tune_lgbm

    ds = settings.datasets.get(dataset)
    fs = feature_set_for(settings, dataset)
    train = prepare_splits(pl.read_parquet(DatasetPaths.of(settings, dataset).features), ds)[
        "train"
    ]
    t = settings.training
    result = tune_lgbm(
        train, fs.names, fs.categorical, ds.splits, t.lgbm, settings.seed,
        trials or t.tune.n_trials, t.categorical_min_count, t.tune.timeout_s,
    )  # fmt: skip
    out = result.save(settings.paths.tuned_params / f"{dataset}.json", fs.spec_hash())
    log.info("best CV PR-AUC %.4f after %d trials -> %s", result.best_value, result.n_trials, out)


def _model_card(settings: Settings, dataset: DatasetName) -> None:
    from fraudshield.training.model_card import write_model_card

    report = settings.paths.reports / dataset / "metrics.json"
    if not report.is_file():
        raise FileNotFoundError(f"{report} not found: run `make train DATASET={dataset}` first")
    log.info("wrote %s", write_model_card(report, settings.paths.model_card))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    dataset: DatasetName = args.dataset or settings.dataset
    try:
        if args.group == "data":
            m = fetch_raw(settings, dataset, force=args.force)
            for f in m.files:
                log.info("%s: %d rows, sha256 %s", f.name, f.rows, f.sha256[:12])
        elif args.group == "features":
            fm = build_features(settings, dataset, check=args.check)
            log.info("%s: %d rows x %d features", dataset, fm.rows, len(fm.feature_names))
        elif args.group == "train":
            _train(settings, dataset, register=not args.no_register)
        elif args.group == "tune":
            _tune(settings, dataset, args.trials)
        elif args.group == "model-card":
            _model_card(settings, dataset)
    except (DownloadError, ReproducibilityError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
