"""The three load-test scenarios (spec: normal, no reason codes, Redis killed)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    explain: bool = True
    # Redis is killed at this share of the run and started again at the second share
    redis_outage: tuple[float, float] | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("normal", "Normal: full scoring with 3 reason codes"),
    Scenario(
        "no_reason_codes", "No reason codes (`explain=false`: no pred_contrib pass)", explain=False
    ),
    Scenario("redis_killed", "Redis killed mid-run, then restarted", redis_outage=(1 / 3, 2 / 3)),
)


def by_name(names: list[str] | None) -> list[Scenario]:
    if not names:
        return list(SCENARIOS)
    known = {s.name: s for s in SCENARIOS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; choose from {sorted(known)}")
    return [known[n] for n in names]
