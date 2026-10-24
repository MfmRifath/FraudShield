"""Online vs offline feature parity check (used by tests and by the promotion gate)."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import polars as pl

from fraudshield.features.definitions import FeatureSet
from fraudshield.features.offline import compute_offline
from fraudshield.features.online import OnlineSpec
from fraudshield.features.store import InMemoryFeatureStore, replay


def max_abs_diff(
    online: list[dict[str, Any]], offline: list[dict[str, Any]], names: list[str]
) -> dict[str, float]:
    """Per-feature max |online - offline|; inf if null-ness or a categorical value disagrees."""
    worst = dict.fromkeys(names, 0.0)
    for a_row, b_row in zip(online, offline, strict=True):
        for n in names:
            a, b = a_row[n], b_row[n]
            if a is None or b is None:
                d = 0.0 if a is None and b is None else math.inf
            elif isinstance(a, str) or isinstance(b, str):
                d = 0.0 if a == b else math.inf
            else:
                d = abs(float(a) - float(b))
            worst[n] = max(worst[n], d)
    return worst


def parity_report(canonical: pl.DataFrame, fs: FeatureSet) -> dict[str, float]:
    """Replay ``canonical`` through the in-memory online path and compare with offline."""
    offline = compute_offline(canonical, fs).select(fs.names).to_dicts()
    store = InMemoryFeatureStore(OnlineSpec.from_feature_set(fs))
    online = asyncio.run(replay(store, canonical.to_dicts(), fs))
    return max_abs_diff(online, offline, fs.names)
