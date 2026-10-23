"""Sparkov credit-card fraud simulation (Kaggle dataset ``kartik2112/fraud-detection``, CC0).

Assumptions (ADR-0004): ``trans_date_trans_time`` is interpreted as UTC wall-clock time
(the dataset's ``unix_time`` column is offset by years and is ignored); ``uid = cc_num``;
``product = category``; ``addr = zip``. Sparkov has no email or device fields, so those
canonical columns are null and the corresponding features are null for this dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

import polars as pl

from fraudshield.adapters.base import DatasetAdapter, extra_column_name

TRAIN_FILE: Final = "fraudTrain.csv"
TEST_FILE: Final = "fraudTest.csv"

_COLUMNS: Final[dict[str, pl.DataType]] = {
    "trans_date_trans_time": pl.String(),
    "cc_num": pl.String(),
    "category": pl.String(),
    "amt": pl.Float64(),
    "zip": pl.String(),
    "trans_num": pl.String(),
    "is_fraud": pl.Int8(),
}


class SparkovAdapter(DatasetAdapter):
    name: ClassVar = "sparkov"
    raw_files: ClassVar = (TRAIN_FILE, TEST_FILE)

    def read_raw(self, files: Sequence[Path]) -> pl.DataFrame:
        extras = list(self.cfg.extra_columns)
        schema = dict(_COLUMNS)
        for c in extras:
            schema.setdefault(c, pl.Float64())
        frames = []
        for f in files:
            header = pl.scan_csv(f).collect_schema().names()
            unknown = sorted(set(schema) - set(header))
            if unknown:
                raise ValueError(f"sparkov: columns missing from {f.name}: {unknown}")
            frames.append(
                pl.scan_csv(f, schema_overrides=schema, infer_schema=False)
                .select(list(schema))
                .collect(engine="in-memory")
            )
        df = pl.concat(frames, how="vertical")
        dupes = df.height - df["trans_num"].n_unique()
        if dupes:
            raise ValueError(f"sparkov: {dupes} duplicate trans_num values across files")
        return df.select(
            txn_id=pl.col("trans_num"),
            event_time=pl.col("trans_date_trans_time").str.to_datetime(
                "%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC"
            ),
            uid=pl.col("cc_num"),
            amount=pl.col("amt"),
            product=pl.col("category"),
            addr=pl.col("zip"),
            label=pl.col("is_fraud"),
            **{extra_column_name(c): pl.col(c) for c in extras},
        )
