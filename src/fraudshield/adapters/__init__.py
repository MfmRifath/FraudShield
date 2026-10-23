"""Dataset adapters mapping raw sources (IEEE-CIS, Sparkov) to the canonical Transaction schema."""

from __future__ import annotations

from fraudshield.adapters.base import DatasetAdapter
from fraudshield.adapters.ieee_cis import IEEECISAdapter
from fraudshield.adapters.sparkov import SparkovAdapter
from fraudshield.config import DatasetName, Settings

ADAPTERS: dict[DatasetName, type[DatasetAdapter]] = {
    "ieee_cis": IEEECISAdapter,
    "sparkov": SparkovAdapter,
}


def get_adapter(name: DatasetName, settings: Settings) -> DatasetAdapter:
    return ADAPTERS[name](settings.datasets.get(name), settings.labels.delay)


__all__ = ["ADAPTERS", "DatasetAdapter", "IEEECISAdapter", "SparkovAdapter", "get_adapter"]
