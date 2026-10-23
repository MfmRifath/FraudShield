"""IEEE-CIS Fraud Detection (Kaggle competition ``ieee-fraud-detection``).

Only the labelled ``train_*`` files are used (the competition test set has no labels).

Assumptions (ADR-0004):

* ``event_time = 2017-12-01T00:00:00Z + TransactionDT seconds``. TransactionDT is a time delta
  from an undisclosed reference; the origin is the community convention, not ground truth.
* ``uid = card1 + "_" + addr1 + "_" + D1n`` with ``D1n = floor(TransactionDT / 86400) - D1``
  (D1 = days since the card's first transaction, so D1n is a per-card constant "start day").
  Missing parts are rendered as ``NA``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

import polars as pl

from fraudshield.adapters.base import DatasetAdapter, extra_column_name

TRANSACTION_FILE: Final = "train_transaction.csv"
IDENTITY_FILE: Final = "train_identity.csv"

_TXN_COLUMNS: Final[dict[str, pl.DataType]] = {
    "TransactionID": pl.Int64(),
    "isFraud": pl.Int8(),
    "TransactionDT": pl.Int64(),
    "TransactionAmt": pl.Float64(),
    "ProductCD": pl.String(),
    "card1": pl.Int64(),
    "card4": pl.String(),
    "card6": pl.String(),
    "addr1": pl.Float64(),
    "P_emaildomain": pl.String(),
    "R_emaildomain": pl.String(),
    "D1": pl.Float64(),
}
_ID_COLUMNS: Final[dict[str, pl.DataType]] = {
    "TransactionID": pl.Int64(),
    "DeviceType": pl.String(),
    "DeviceInfo": pl.String(),
}


def _int_str(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64).cast(pl.String).fill_null("NA")


def uid_expr() -> pl.Expr:
    d1n = (pl.col("TransactionDT") // 86_400) - pl.col("D1")
    return pl.concat_str(
        [_int_str("card1"), _int_str("addr1"), d1n.cast(pl.Int64).cast(pl.String).fill_null("NA")],
        separator="_",
    )


class IEEECISAdapter(DatasetAdapter):
    name: ClassVar = "ieee_cis"
    raw_files: ClassVar = (TRANSACTION_FILE, IDENTITY_FILE)

    def read_raw(self, files: Sequence[Path]) -> pl.DataFrame:
        by_name = {f.name: f for f in files}
        txn_path, id_path = by_name[TRANSACTION_FILE], by_name[IDENTITY_FILE]

        header = pl.scan_csv(txn_path).collect_schema().names()
        extras = list(self.cfg.extra_columns)
        unknown = sorted(set(extras) - set(header))
        if unknown:
            raise ValueError(f"ieee_cis extra_columns not in {TRANSACTION_FILE}: {unknown}")
        txn_schema = dict(_TXN_COLUMNS)
        for c in extras:
            txn_schema.setdefault(c, pl.Float64())
        txn = (
            pl.scan_csv(txn_path, schema_overrides=txn_schema, infer_schema=False)
            .select(list(txn_schema))
            .collect(engine="in-memory")
        )
        ident = (
            pl.scan_csv(id_path, schema_overrides=_ID_COLUMNS, infer_schema=False)
            .select(list(_ID_COLUMNS))
            .collect(engine="in-memory")
        )
        df = txn.join(ident, on="TransactionID", how="left", validate="1:1", maintain_order="left")

        origin = pl.lit(self.cfg.origin).dt.cast_time_unit("us")
        return df.select(
            txn_id=pl.col("TransactionID").cast(pl.String),
            event_time=origin + pl.duration(seconds=pl.col("TransactionDT")),
            uid=uid_expr(),
            amount=pl.col("TransactionAmt"),
            product=pl.col("ProductCD"),
            card_network=pl.col("card4"),
            card_kind=pl.col("card6"),
            p_email_domain=pl.col("P_emaildomain"),
            r_email_domain=pl.col("R_emaildomain"),
            device_type=pl.col("DeviceType"),
            device_info=pl.col("DeviceInfo"),
            addr=pl.col("addr1").cast(pl.Int64).cast(pl.String),
            label=pl.col("isFraud"),
            **{extra_column_name(c): pl.col(c).cast(pl.Float64) for c in extras},
        )
