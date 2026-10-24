"""Failure-rate circuit breaker for the feature-store call.

closed --(>= min_calls in the last window_s and failure share >= failure_ratio)--> open
open --(reset_s elapsed)--> half-open (one trial call)
half-open --success--> closed ; half-open --failure--> open for another reset_s

Rate-based, not "N consecutive failures": timeouts are not independent. One event-loop or VM
stall times out every fetch in flight at once, and a consecutive-failure breaker turned each such
hiccup into reset_s of degraded scoring (found by the Phase 8 load test, ADR-0014). A dead store
still fails every call, so the breaker opens after ``min_calls`` requests.

A trial that ends without a verdict (the request was cancelled) frees the trial slot (``abandon``),
otherwise the breaker would stay open forever.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_ratio: float = 0.5,
        min_calls: int = 10,
        window_s: float = 2.0,
        reset_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < failure_ratio <= 1.0:
            raise ValueError(f"failure_ratio must be in (0, 1], got {failure_ratio}")
        self.failure_ratio = failure_ratio
        self.min_calls = min_calls
        self.window_s = window_s
        self.reset_s = reset_s
        self._clock = clock
        self._calls: deque[tuple[float, bool]] = deque()  # (time, failed) inside the window
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
        if self._opened_at is not None:  # successful trial: close with a clean window
            self._close()
            return
        self._record(failed=False)

    def failure(self) -> None:
        self._trial_in_flight = False
        if self._opened_at is not None:  # failed trial: stay open for another period
            self._opened_at = self._clock()
            return
        self._record(failed=True)
        n = len(self._calls)
        if n >= self.min_calls and self._failures >= self.failure_ratio * n:
            self._opened_at = self._clock()
            self._calls.clear()
            self._failures = 0

    def abandon(self) -> None:
        """The call ended without success or failure (e.g. cancelled): allow another trial."""
        self._trial_in_flight = False

    def _close(self) -> None:
        self._opened_at = None
        self._trial_in_flight = False
        self._calls.clear()
        self._failures = 0

    def _record(self, *, failed: bool) -> None:
        now = self._clock()
        self._calls.append((now, failed))
        self._failures += failed
        horizon = now - self.window_s
        while self._calls and self._calls[0][0] <= horizon:
            _, old_failed = self._calls.popleft()
            self._failures -= old_failed
