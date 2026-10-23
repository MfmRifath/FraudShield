"""Shared pytest configuration.

Hypothesis profiles: ``dev`` (default, fast) and ``ci`` (more examples, derandomized).
Select with ``HYPOTHESIS_PROFILE=ci``.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

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
