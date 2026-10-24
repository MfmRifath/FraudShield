"""``fraudshield`` command-line entry point (wired to Makefile targets)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
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

    rt = with_dataset(
        sub.add_parser("retrain", help="training set as of now -> train -> @challenger")
    )
    rt.add_argument(
        "--as-of", type=datetime.fromisoformat, default=None, help="label cutoff (UTC ISO)"
    )
    pr = with_dataset(
        sub.add_parser("promote", help="gate @challenger vs @champion; promote if it passes")
    )
    pr.add_argument(
        "--force", action="store_true", help="promote even if the gate fails (recorded)"
    )
    sub.add_parser("rollback", help="move @champion back to the previous version")
    cn = sub.add_parser("canary", help="set the API's canary share for @challenger")
    cn.add_argument("pct", type=int)
    cn.add_argument("--api-url", default=None, help="defaults to mlops.api_url / streaming.api_url")
    md = sub.add_parser(
        "mlops-demo", help="retrain -> challenger -> gate/shadow -> promote -> rollback (synthetic)"
    )
    md.add_argument("--workdir", type=Path, default=Path("artifacts/mlops-demo"))
    md.add_argument(
        "--tracking-uri", default=None, help="MLflow server (default: SQLite in workdir)"
    )

    rp = with_dataset(
        sub.add_parser("replay", help="replay transactions into Kafka (transactions.raw/.truth)")
    )
    rp.add_argument("--source", choices=("fixture", "canonical"), default="fixture",
                    help="fixture = committed CC0 Sparkov sample; canonical = data/interim/<ds>")  # fmt: skip
    rp.add_argument(
        "--start", type=datetime.fromisoformat, default=None, help="UTC ISO time (inclusive)"
    )
    rp.add_argument(
        "--end", type=datetime.fromisoformat, default=None, help="UTC ISO time (exclusive)"
    )
    rp.add_argument("--max-events", type=int, default=None)
    rp.add_argument(
        "--speed", type=float, default=0.0, help="event-time seconds per wall second (0 = off)"
    )
    rp.add_argument(
        "--rate", type=float, default=0.0, help="max events per wall second (0 = unlimited)"
    )
    rp.add_argument("--namespace", default=None, help="prefix ids/entities to isolate a run")
    rp.add_argument("--drift", type=float, default=0.0, metavar="SHARE",
                    help="inject a fraud ring into this share of events (drift demo)")  # fmt: skip
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


def _replay(settings: Settings, dataset: DatasetName, args: argparse.Namespace) -> None:
    from confluent_kafka import Producer

    from fraudshield.streaming.replayer import load_source, replay, select_window

    df = select_window(
        load_source(settings, args.source, dataset), args.start, args.end, args.max_events
    )
    producer = Producer({"bootstrap.servers": settings.streaming.bootstrap_servers, "linger.ms": 5})
    stats = replay(
        df.to_dicts(),
        producer,
        settings,
        speed=args.speed,
        rate=args.rate,
        namespace=args.namespace,
        drift_share=args.drift,
    )
    log.info("replayed %d transactions, %d truth records", stats.raw, stats.truth)


def _mlops(settings: Settings, dataset: DatasetName, args: argparse.Namespace) -> None:
    import json

    import requests

    from fraudshield.mlops.loop import promote, retrain, rollback

    if args.group == "retrain":
        result, stats = retrain(settings, dataset, as_of=args.as_of)
        log.info("v%s registered as @challenger (training set %s)", result.model_version, stats)
    elif args.group == "promote":
        out = promote(settings, dataset, force=args.force)
        log.info(
            "gate %s; promoted=%s; report %s",
            "passed" if out.gate.passed else "FAILED",
            out.promoted,
            out.report,
        )
    elif args.group == "rollback":
        move = rollback(settings)
        log.info("@%s rolled back: v%s -> v%s", move.alias, move.before, move.after)
    elif args.group == "canary":
        url = (args.api_url or settings.mlops.api_url or settings.streaming.api_url).rstrip("/")
        token = settings.serving.admin_token
        if token is None:
            raise FileNotFoundError("set FRAUDSHIELD_SERVING__ADMIN_TOKEN to change the canary")
        r = requests.post(f"{url}/admin/canary", json={"pct": args.pct},
                          headers={"X-Admin-Token": token.get_secret_value()}, timeout=10)  # fmt: skip
        log.info("%d %s", r.status_code, r.text)
    elif args.group == "mlops-demo":
        from fraudshield.mlops.demo import mlops_demo

        print(
            json.dumps(mlops_demo(settings, args.workdir, args.tracking_uri), indent=2, default=str)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    dataset: DatasetName = getattr(args, "dataset", None) or settings.dataset
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
        elif args.group == "replay":
            _replay(settings, dataset, args)
        elif args.group in ("retrain", "promote", "rollback", "canary", "mlops-demo"):
            _mlops(settings, dataset, args)
    except (DownloadError, ReproducibilityError, FileNotFoundError, RuntimeError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
