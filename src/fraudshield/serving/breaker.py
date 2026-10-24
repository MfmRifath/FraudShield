"""Minimal circuit breaker for the feature-store call.

closed --(N consecutive failures)--> open --(reset_s elapsed)--> half-open (one trial call)
half-open --success--> closed ; half-open --failure--> open
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(
        self, failure_threshold: int, reset_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    def failure(self) -> None:
        self._trial_in_flight = False
        if self._opened_at is not None:  # failed trial: stay open for another period
            self._opened_at = self._clock()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
