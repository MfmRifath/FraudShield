"""Shared pytest configuration.

Hypothesis profiles: ``dev`` (default, fast) and ``ci`` (more examples, derandomized).
Select with ``HYPOTHESIS_PROFILE=ci``.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

try:
    import lightgbm  # noqa: F401
except OSError as e:  # pragma: no cover - environment specific
    if "libomp" in str(e):
        raise pytest.UsageError(
            "LightGBM cannot load the OpenMP runtime (libomp). On macOS run `brew install libomp`, "
            "or run the suite in Linux with `make test-docker`."
        ) from e
    raise

settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci",
    max_examples=300,
    deadline=None,
    derandomize=True,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
