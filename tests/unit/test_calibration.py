from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.isotonic import IsotonicRegression

from fraudshield.training.calibrate import Calibrator, fit_calibrator


def _data(n: int, pos_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    s = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    return s, y


@pytest.mark.unit
def test_isotonic_when_enough_positives_matches_sklearn() -> None:
    s, y = _data(20_000, 0.2, 0)
    cal = fit_calibrator(s, y, platt_below_positives=1000)
    assert cal.kind == "isotonic"
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    grid = np.linspace(-0.5, 1.5, 501)
    np.testing.assert_allclose(cal.apply(grid), ref.predict(grid), atol=1e-12)


@pytest.mark.unit
def test_platt_when_few_positives() -> None:
    s, y = _data(3_000, 0.05, 1)
    assert y.sum() < 1000
    cal = fit_calibrator(s, y, platt_below_positives=1000)
    assert cal.kind == "platt"
    assert cal.a > 0


@pytest.mark.unit
def test_single_class_falls_back_to_identity() -> None:
    cal = fit_calibrator(np.array([0.2, 0.4]), np.array([0, 0]))
    assert cal.kind == "identity"
    np.testing.assert_allclose(cal.apply([0.2, 1.7]), [0.2, 1.0])


@pytest.mark.unit
@pytest.mark.parametrize(("n", "rate", "seed"), [(20_000, 0.2, 2), (3_000, 0.05, 3)])
@given(grid=hnp.arrays(np.float64, st.integers(2, 200), elements=st.floats(0, 1)))
def test_calibration_is_monotone_and_bounded(
    n: int, rate: float, seed: int, grid: np.ndarray
) -> None:
    s, y = _data(n, rate, seed)
    cal = fit_calibrator(s, y)
    g = np.sort(grid)
    out = cal.apply(g)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1))


@pytest.mark.unit
def test_json_roundtrip() -> None:
    s, y = _data(20_000, 0.2, 4)
    cal = fit_calibrator(s, y)
    back = Calibrator.model_validate_json(cal.model_dump_json())
    np.testing.assert_array_equal(back.apply(s), cal.apply(s))
