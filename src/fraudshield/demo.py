"""Single-container demo: the scoring API, an in-process replayer and the analyst UI.

``uvicorn --factory fraudshield.demo:create_demo_app`` serves:

* ``/api/*``  the regular API (``serving.app``) with the in-memory feature store, in-memory case
  store, inline store updates and no broker (``publisher=none``);
* ``/*``      the built UI (``ui/dist``), with index.html for client-side routes;
* a background task replaying a window of transactions through the scorer at ``demo.rate`` per
  second, releasing each transaction's chargeback label when its simulated availability time
  passes (same rule as the streaming label emitter). With ``demo.loop`` it starts again with
  event times shifted past the window, so the cards keep their history.

Everything that grows is bounded (case store cap, history retention, no published-message
buffer), so it runs indefinitely in well under 1 GB (``make demo-check``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

from fraudshield.config import Settings, get_settings
from fraudshield.serving.app import create_app
from fraudshield.serving.schemas import ScoreRequest
from fraudshield.streaming.common import parse_time
from fraudshield.streaming.label_emitter import LabelScheduler
from fraudshield.streaming.replayer import load_source, raw_message

log = logging.getLogger(__name__)


def load_window(settings: Settings) -> pl.DataFrame:
    """The replay window, in event order: ``demo.window`` (canonical parquet) or the fixture."""
    w = settings.demo.window
    df = pl.read_parquet(w) if w is not None else load_source(settings, "fixture", "sparkov")
    return df.sort("event_time", "txn_id")


async def replay(
    api: FastAPI, settings: Settings, *, max_events: int | None = None, pace: bool = True
) -> int:
    """Score the window through the in-process service; returns the number of events scored."""
    svc = api.state.fs.service
    store = svc.store
    rows = load_window(settings).to_dicts()
    if not rows:
        return 0
    span = rows[-1]["event_time"] - rows[0]["event_time"] + timedelta(days=1)
    attrs = store.spec.te_columns
    labels = LabelScheduler(settings.labels.delay)
    n, loop, delay = 0, 0, 1.0 / settings.demo.rate
    clock = asyncio.get_running_loop().time
    next_at = clock()
    while True:
        shift = span * loop
        for row in rows:
            t = row["event_time"] + shift
            body: dict[str, Any] = raw_message({**row, "event_time": t})
            body["txn_id"] = f"{row['txn_id']}~{loop}" if loop else row["txn_id"]
            await svc.score(ScoreRequest.model_validate(body))
            if row.get("label") is not None:
                truth = {"txn_id": body["txn_id"], "uid": row["uid"], "event_time": t.isoformat(),
                         "label": row["label"], "attrs": {c: row.get(c) for c in attrs}}  # fmt: skip
                for lab in labels.on_truth(truth):
                    lat = parse_time(lab["label_available_time"])
                    if lat is not None:
                        await store.add_label(lab["txn_id"], lab["label"], lat, lab["attrs"])
            n += 1
            if max_events is not None and n >= max_events:
                return n
            if pace:  # absolute schedule: scoring time does not slow the replay rate
                next_at = max(next_at + delay, clock() - 1.0)  # no catch-up burst after a stall
                await asyncio.sleep(max(0.0, next_at - clock()))
        loop += 1
        if not settings.demo.loop:
            return n


def create_demo_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    api = create_app(s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Starlette does not run a mounted app's lifespan: enter it here, then start replaying
        async with api.router.lifespan_context(api):
            task = asyncio.create_task(_replay_logged(api, s), name="demo-replayer")
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="FraudShield demo", lifespan=lifespan, openapi_url=None)
    app.mount("/api", api)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    ui = s.demo.ui_dir.resolve()

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def ui_files(path: str) -> FileResponse | PlainTextResponse:
        f = (ui / path).resolve()
        if path and f.is_file() and f.is_relative_to(ui):  # no traversal out of the UI build
            return FileResponse(f)
        index = ui / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return PlainTextResponse("UI not built (cd ui && npm run build)", status_code=404)

    return app


async def _replay_logged(api: FastAPI, settings: Settings) -> None:
    try:
        n = await replay(api, settings)
        log.info("demo replay finished after %d events", n)
    except asyncio.CancelledError:
        raise
    except Exception:  # keep serving the API and UI even if the replayer dies
        log.exception("demo replay stopped")


def write_demo_window(dataset: str, max_rows: int, out: Path) -> int:
    """The first ``max_rows`` events of a dataset's test window, for ``demo.window``.

    The demo keeps its window in memory, so it is a slice: the whole Sparkov test split would
    not fit the 1 GB budget.
    """
    from fraudshield.pipeline import DatasetPaths
    from fraudshield.training.splits import assign_splits

    s = get_settings()
    ds = s.datasets.get(dataset)  # type: ignore[arg-type]
    canon = pl.read_parquet(DatasetPaths.of(s, dataset).canonical)  # type: ignore[arg-type]
    test = assign_splits(canon, ds.origin, ds.splits).filter(pl.col("split") == "test")
    window = test.drop("day", "split").sort("event_time", "txn_id").head(max_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    window.write_parquet(out)
    return window.height


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m fraudshield.demo")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window", help="write a test-window slice for the demo replayer")
    w.add_argument("--dataset", default="sparkov", choices=["ieee_cis", "sparkov"])
    w.add_argument("--max-rows", type=int, default=20_000)
    w.add_argument("--out", type=Path, default=Path("data/processed/sparkov/demo_window.parquet"))
    args = p.parse_args(argv)
    n = write_demo_window(args.dataset, args.max_rows, args.out)
    print(f"{n:,} events -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
