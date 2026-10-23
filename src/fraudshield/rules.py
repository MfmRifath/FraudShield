"""Hard rules from ``configs/rules.yaml``.

Used twice with identical semantics: per event in serving (``RuleSet.evaluate_row``) and
vectorised for the rules-only baseline (``RuleSet.evaluate_frame``). A rule fires when *all* of
its conditions hold; a condition on a null/missing feature never holds. Block beats review.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

Op = Literal["gt", "ge", "lt", "le", "eq", "ne", "in"]
_SCALAR_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Condition(_Model):
    feature: str
    op: Op
    value: float | str | tuple[float | str, ...]

    def holds(self, row: Mapping[str, Any]) -> bool:
        v = row.get(self.feature)
        if v is None or (isinstance(v, float) and v != v):  # null or NaN
            return False
        if self.op == "in":
            return v in (self.value if isinstance(self.value, tuple) else (self.value,))
        return bool(_SCALAR_OPS[self.op](v, self.value))

    def expr(self) -> pl.Expr:
        col = pl.col(self.feature)
        if self.op == "in":
            values = list(self.value) if isinstance(self.value, tuple) else [self.value]
            e = col.is_in(values)
        else:
            e = getattr(col, f"__{self.op}__")(self.value)
        return (
            col.is_not_null() & ~col.is_nan() if _is_float(self.value) else col.is_not_null()
        ) & e


def _is_float(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


class Rule(_Model):
    name: str
    action: Literal["block", "review"]
    reason: str
    all: tuple[Condition, ...] = Field(min_length=1)

    def fires(self, row: Mapping[str, Any]) -> bool:
        return all(c.holds(row) for c in self.all)

    def expr(self) -> pl.Expr:
        e = self.all[0].expr()
        for c in self.all[1:]:
            e = e & c.expr()
        return e.fill_null(False)


@dataclass(frozen=True, slots=True)
class RuleResult:
    action: Literal["block", "review"] | None
    fired: tuple[str, ...]
    reasons: tuple[str, ...]


class RuleSet(_Model):
    rules: tuple[Rule, ...] = ()

    @classmethod
    def load(cls, path: Path) -> RuleSet:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})

    @property
    def features(self) -> set[str]:
        return {c.feature for r in self.rules for c in r.all}

    def evaluate_row(self, row: Mapping[str, Any]) -> RuleResult:
        fired = [r for r in self.rules if r.fires(row)]
        action: Literal["block", "review"] | None = None
        if any(r.action == "block" for r in fired):
            action = "block"
        elif fired:
            action = "review"
        return RuleResult(action, tuple(r.name for r in fired), tuple(r.reason for r in fired))

    def evaluate_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """Columns ``rule_action`` (block/review/null) and ``rules_fired`` (list of names)."""
        if not self.rules:
            return pl.DataFrame(
                {"rule_action": [None] * df.height, "rules_fired": [[]] * df.height},
                schema={"rule_action": pl.String, "rules_fired": pl.List(pl.String)},
            )
        flags = df.select([r.expr().alias(r.name) for r in self.rules])
        block = pl.any_horizontal(
            [pl.col(r.name) for r in self.rules if r.action == "block"] or [pl.lit(False)]
        )
        review = pl.any_horizontal(
            [pl.col(r.name) for r in self.rules if r.action == "review"] or [pl.lit(False)]
        )
        fired = pl.concat_list(
            [pl.when(pl.col(r.name)).then(pl.lit(r.name)) for r in self.rules]
        ).list.drop_nulls()
        return flags.select(
            rule_action=pl.when(block).then(pl.lit("block")).when(review).then(pl.lit("review")),
            rules_fired=fired,
        )
