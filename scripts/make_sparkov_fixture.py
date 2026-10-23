"""Generate tests/fixtures/sparkov_sample.csv: 2,000 raw Sparkov rows (CC0) for tests.

Deterministic: every card active in a 3-week window is ranked by a stable hash of its cc_num;
all cards with fraud in the window come first (capped), then non-fraud cards, until at least
N_ROWS rows are collected; the result is cut to the first N_ROWS rows in time order (a time
prefix, so point-in-time features of the kept rows are unaffected by the cut).

The CSV keeps the exact raw file format (all columns, original strings), so it exercises the
real SparkovAdapter.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import polars as pl

RAW = Path("data/raw/sparkov/fraudTrain.csv")
OUT = Path("tests/fixtures/sparkov_sample.csv")
WINDOW = ("2019-03-01 00:00:00", "2019-03-22 00:00:00")
N_ROWS = 2_000
MAX_FRAUD_CARDS = 6


def _rank(cc: str) -> str:
    return hashlib.sha256(cc.encode()).hexdigest()


def main() -> int:
    if not RAW.is_file():
        print(f"{RAW} missing: run `make data DATASET=sparkov` first", file=sys.stderr)
        return 1
    raw = pl.read_csv(RAW, infer_schema=False)  # every column as the original string
    lo, hi = (pl.lit(w) for w in WINDOW)
    win = raw.filter(pl.col("trans_date_trans_time").is_between(lo, hi, closed="left"))
    cards = (
        win.group_by("cc_num")
        .agg(n=pl.len(), fraud=(pl.col("is_fraud") == "1").any())
        .with_columns(rank=pl.col("cc_num").map_elements(_rank, return_dtype=pl.String))
        .sort("rank")
    )
    ordered = pl.concat(
        [cards.filter(pl.col("fraud")).head(MAX_FRAUD_CARDS), cards.filter(~pl.col("fraud"))]
    )
    chosen: list[str] = []
    total = 0
    for cc, n in ordered.select("cc_num", "n").iter_rows():
        chosen.append(cc)
        total += n
        if total >= N_ROWS * 1.15:  # headroom so the time-prefix cut still keeps every card
            break
    sample = (
        win.filter(pl.col("cc_num").is_in(chosen))
        .sort("trans_date_trans_time", "trans_num", maintain_order=True)
        .head(N_ROWS)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(OUT)
    frauds = (sample["is_fraud"] == "1").sum()
    print(f"wrote {OUT}: {sample.height} rows, {len(chosen)} cards, {frauds} fraud rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
