from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fraudshield.adapters import IEEECISAdapter, SparkovAdapter, get_adapter
from fraudshield.config import Settings
from fraudshield.schema import CANONICAL_COLUMNS

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sparkov_sample.csv"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def sparkov(settings: Settings) -> pl.DataFrame:
    return get_adapter("sparkov", settings).load([FIXTURE])


# ------------------------------------------------------------------ Sparkov (real CC0 sample)
@pytest.mark.unit
def test_sparkov_fixture_maps_to_canonical(sparkov: pl.DataFrame) -> None:
    raw = pl.read_csv(FIXTURE, infer_schema=False)
    assert sparkov.height == raw.height == 2_000
    assert sparkov.columns[: len(CANONICAL_COLUMNS)] == list(CANONICAL_COLUMNS)
    assert [c for c in sparkov.columns if c.startswith("x_")] == [
        "x_city_pop",
        "x_lat",
        "x_long",
        "x_merch_lat",
        "x_merch_long",
    ]
    assert set(sparkov["uid"]) == set(raw["cc_num"])
    assert set(sparkov["txn_id"]) == set(raw["trans_num"])
    assert int(sparkov["label"].sum()) == (raw["is_fraud"] == "1").sum() > 0
    for col in ("p_email_domain", "device_info", "card_network"):
        assert sparkov[col].null_count() == sparkov.height  # not present in Sparkov


@pytest.mark.unit
def test_sparkov_times_are_utc_wallclock(sparkov: pl.DataFrame) -> None:
    raw = pl.read_csv(FIXTURE, infer_schema=False)
    first = raw.sort("trans_date_trans_time", "trans_num").row(0, named=True)
    row = sparkov.filter(pl.col("txn_id") == first["trans_num"]).row(0, named=True)
    parsed = datetime.strptime(first["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=UTC
    )
    assert abs(row["event_time"] - parsed) < timedelta(seconds=1)  # tie-break adds only µs
    assert row["amount"] == float(first["amt"])


@pytest.mark.unit
def test_sparkov_rejects_duplicate_trans_num(settings: Settings, tmp_path: Path) -> None:
    dup = tmp_path / "fraudTest.csv"
    pl.read_csv(FIXTURE, infer_schema=False).head(3).write_csv(dup)
    with pytest.raises(ValueError, match="duplicate trans_num"):
        SparkovAdapter(settings.datasets.sparkov, settings.labels.delay).load([FIXTURE, dup])


@pytest.mark.unit
def test_missing_raw_file_has_helpful_error(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        get_adapter("sparkov", settings).load([tmp_path / "nope.csv"])


# ------------------------------------------------------------------ IEEE-CIS (synthetic rows)
# Hand-written rows in the IEEE-CIS file layout (the real data is not redistributable).
def _write_ieee(tmp_path: Path, extra_cols: tuple[str, ...]) -> list[Path]:
    txn = pl.DataFrame(
        {
            "TransactionID": [2987000, 2987001, 2987002, 2987003],
            "isFraud": [0, 1, 0, 0],
            "TransactionDT": [86_400, 86_401, 86_400 * 5 + 30, 86_400],
            "TransactionAmt": [68.5, 29.0, 59.0, 50.0],
            "ProductCD": ["W", "W", "C", "W"],
            "card1": [13926, 2755, 13926, 4663],
            "card2": [None, 404.0, None, 490.0],
            "card3": [150.0, 150.0, 150.0, 150.0],
            "card4": ["discover", "mastercard", "discover", "visa"],
            "card5": [142.0, 102.0, 142.0, 166.0],
            "card6": ["credit", "credit", "credit", "debit"],
            "addr1": [315.0, 325.0, 315.0, None],
            "addr2": [87.0, 87.0, 87.0, None],
            "dist1": [19.0, None, None, None],
            "P_emaildomain": [None, "gmail.com", "outlook.com", "yahoo.com"],
            "R_emaildomain": [None, None, "outlook.com", None],
            "C1": [1.0, 1.0, 2.0, 2.0],
            "D1": [14.0, 0.0, 18.0, None],
        }
    )
    ident = pl.DataFrame(
        {
            "TransactionID": [2987001, 2987002],
            "id_01": [0.0, -5.0],
            "DeviceType": ["mobile", "desktop"],
            "DeviceInfo": ["SAMSUNG SM-G892A Build/NRD90M", "Windows"],
        }
    )
    t, i = tmp_path / "train_transaction.csv", tmp_path / "train_identity.csv"
    txn.select([c for c in txn.columns if c in {*txn.columns[:16], *extra_cols, "D1"}]).write_csv(t)
    ident.write_csv(i)
    return [t, i]


@pytest.fixture
def ieee_settings() -> Settings:
    s = Settings()
    ds = s.datasets.ieee_cis.model_copy(update={"extra_columns": ("card2", "C1", "D1")})
    return s.model_copy(update={"datasets": s.datasets.model_copy(update={"ieee_cis": ds})})


@pytest.mark.unit
def test_ieee_uid_time_identity_join(ieee_settings: Settings, tmp_path: Path) -> None:
    df = get_adapter("ieee_cis", ieee_settings).load(_write_ieee(tmp_path, ("card2", "C1")))
    rows = {r["txn_id"]: r for r in df.to_dicts()}
    origin = datetime(2017, 12, 1, tzinfo=UTC)
    # D1n = floor(DT / 86400) - D1 ; missing parts become NA
    assert rows["2987000"]["uid"] == "13926_315_-13"  # day 1 - 14
    assert rows["2987002"]["uid"] == "13926_315_-13"  # day 5 - 18 -> same card, same uid
    assert rows["2987001"]["uid"] == "2755_325_1"
    assert rows["2987003"]["uid"] == "4663_NA_NA"
    # TransactionDT 86400 occurs twice: tie broken by txn_id, +1 µs for the later one
    assert rows["2987000"]["event_time"] == origin + timedelta(seconds=86_400)
    assert rows["2987003"]["event_time"] == origin + timedelta(seconds=86_400, microseconds=1)
    assert rows["2987001"]["device_type"] == "mobile"
    assert rows["2987000"]["device_type"] is None  # no identity row -> left join null
    assert rows["2987001"]["card_network"] == "mastercard"
    assert rows["2987003"]["card_kind"] == "debit"
    assert rows["2987000"]["addr"] == "315"
    assert rows["2987001"]["label"] == 1
    assert rows["2987000"]["x_card2"] is None
    assert rows["2987001"]["x_c1"] == 1.0
    assert rows["2987002"]["x_d1"] == 18.0


@pytest.mark.unit
def test_ieee_unknown_extra_column_is_rejected(ieee_settings: Settings, tmp_path: Path) -> None:
    ds = ieee_settings.datasets.ieee_cis.model_copy(update={"extra_columns": ("V999",)})
    adapter = IEEECISAdapter(ds, ieee_settings.labels.delay)
    with pytest.raises(ValueError, match="V999"):
        adapter.load(_write_ieee(tmp_path, ()))
