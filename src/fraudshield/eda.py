"""Exploratory summaries over a canonical frame (used by notebooks/01_eda.ipynb)."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from fraudshield.training.splits import day_number


def daily_class_balance(df: pl.DataFrame, origin: datetime) -> pl.DataFrame:
    """Per dataset day: transactions, labelled frauds and fraud rate."""
    return (
        df.group_by(day_number(origin).alias("day"))
        .agg(n=pl.len(), n_fraud=pl.col("label").sum().cast(pl.Int64))
        .with_columns(fraud_rate=pl.col("n_fraud") / pl.col("n"))
        .sort("day")
    )


def amount_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Amount distribution by label."""
    a = pl.col("amount")
    return (
        df.group_by("label")
        .agg(
            n=pl.len(),
            mean=a.mean(),
            p50=a.quantile(0.5, "nearest"),
            p90=a.quantile(0.9, "nearest"),
            p99=a.quantile(0.99, "nearest"),
            max=a.max(),
        )
        .sort("label", nulls_last=True)
    )


def uid_summary(df: pl.DataFrame) -> pl.DataFrame:
    """One-row summary of entity (uid) statistics."""
    per_uid = df.group_by("uid").agg(n=pl.len(), n_fraud=pl.col("label").sum())
    n = pl.col("n")
    return per_uid.select(
        n_uids=pl.len(),
        txns_per_uid_p50=n.quantile(0.5, "nearest"),
        txns_per_uid_p90=n.quantile(0.9, "nearest"),
        txns_per_uid_max=n.max(),
        share_uids_with_fraud=(pl.col("n_fraud") > 0).mean(),
        single_txn_uid_share=(n == 1).mean(),
    )


def _is_new_uid() -> pl.Expr:
    return pl.int_range(pl.len()).over("uid") == 0  # canonical frames are time-sorted


def new_vs_known_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Pooled counts and fraud rates for a uid's first-ever event vs later events."""
    return (
        df.with_columns(
            segment=pl.when(_is_new_uid()).then(pl.lit("new")).otherwise(pl.lit("known"))
        )
        .group_by("segment")
        .agg(n=pl.len(), n_fraud=pl.col("label").sum().cast(pl.Int64))
        .with_columns(
            share=pl.col("n") / pl.col("n").sum(), fraud_rate=pl.col("n_fraud") / pl.col("n")
        )
        .sort("segment", descending=True)
    )


def new_uid_share_by_day(df: pl.DataFrame, origin: datetime) -> pl.DataFrame:
    """Per day: share of transactions that are a uid's first-ever event, and their fraud rate."""
    first = _is_new_uid()
    return (
        df.with_columns(day=day_number(origin), is_new=first)
        .group_by("day")
        .agg(
            n=pl.len(),
            new_share=pl.col("is_new").mean(),
            fraud_rate_new=pl.col("label").filter(pl.col("is_new")).mean(),
            fraud_rate_known=pl.col("label").filter(~pl.col("is_new")).mean(),
        )
        .sort("day")
    )
