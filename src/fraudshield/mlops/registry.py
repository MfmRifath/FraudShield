"""MLflow registry aliases with a promotion history (so rollback is one command).

MLflow stages are deprecated; we only move aliases. Every promotion appends the previous
``@champion`` version to the registered-model tag ``champion_history`` (JSON list), and
``rollback`` pops it back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from mlflow.exceptions import MlflowException

from fraudshield.config import Settings
from fraudshield.mlflow_utils import client as mlflow_client
from fraudshield.mlflow_utils import download_model_version
from fraudshield.model_bundle import ModelBundle, find_bundle_dir

log = logging.getLogger(__name__)
HISTORY_TAG = "champion_history"


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class AliasMove:
    alias: str
    before: str | None
    after: str


class Registry:
    def __init__(
        self,
        tracking_uri: str,
        name: str,
        champion: str = "champion",
        challenger: str = "challenger",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.name = name
        self.champion = champion
        self.challenger = challenger
        self.client = mlflow_client(tracking_uri)

    @classmethod
    def from_settings(cls, settings: Settings) -> Registry:
        m = settings.mlops
        return cls(
            settings.training.mlflow_tracking_uri,
            settings.training.registered_model,
            m.champion_alias,
            m.challenger_alias,
        )

    def version_of(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.name, alias).version)
        except MlflowException:
            return None

    def history(self) -> list[str]:
        try:
            tags = self.client.get_registered_model(self.name).tags
        except MlflowException:
            return []
        return list(json.loads(tags.get(HISTORY_TAG, "[]")))

    def _set_history(self, history: list[str]) -> None:
        self.client.set_registered_model_tag(self.name, HISTORY_TAG, json.dumps(history))

    def promote(self, version: str) -> AliasMove:
        before = self.version_of(self.champion)
        if before == version:
            raise RegistryError(f"version {version} is already @{self.champion}")
        if before is not None:
            self._set_history([*self.history(), before])
        self.client.set_registered_model_alias(self.name, self.champion, version)
        self.client.set_model_version_tag(
            self.name, version, "promoted_at", datetime.now(UTC).isoformat(timespec="seconds")
        )
        log.info("@%s: %s -> %s", self.champion, before, version)
        return AliasMove(self.champion, before, version)

    def rollback(self) -> AliasMove:
        history = self.history()
        if not history:
            raise RegistryError("no previous champion to roll back to")
        target = history.pop()
        before = self.version_of(self.champion)
        self.client.set_registered_model_alias(self.name, self.champion, target)
        self._set_history(history)
        if before is not None:
            self.client.set_model_version_tag(
                self.name, before, "rolled_back_at", datetime.now(UTC).isoformat(timespec="seconds")
            )
        log.info("rollback @%s: %s -> %s", self.champion, before, target)
        return AliasMove(self.champion, before, target)

    def load_bundle(self, version: str) -> ModelBundle:
        return ModelBundle.load(
            find_bundle_dir(download_model_version(self.tracking_uri, self.name, version))
        )
