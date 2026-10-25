"""Model slots (@champion / @challenger), canary routing and atomic hot reload."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from fraudshield.config import Settings
from fraudshield.features.definitions import FeatureSet
from fraudshield.model_bundle import ModelBundle, find_bundle_dir
from fraudshield.pipeline import feature_set_for

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSlot:
    alias: str
    version: str
    bundle: ModelBundle


def canary_bucket(uid: str) -> int:
    """Stable 0..99 bucket for a uid (independent of Python's per-process hash seed)."""
    return int.from_bytes(hashlib.blake2b(uid.encode(), digest_size=8).digest(), "big") % 100


def _load_bundle_dir(path: Path, alias: str) -> ModelSlot:
    if not (path / "bundle.json").is_file():
        raise ModelLoadError(f"no model bundle at {path}")
    bundle = ModelBundle.load(path)
    version = str(
        bundle.spec.metadata.get("model_version")
        or f"bundle:{bundle.spec.metadata.get('created_at', 'unknown')}"
    )
    return ModelSlot(alias, version, bundle)


def _load_mlflow(uri: str, alias: str, tracking_uri: str) -> ModelSlot:
    """``models:/<name>@<alias>`` -> download the pyfunc and load its bundle artifact directly."""
    from fraudshield.mlflow_utils import client, download_model_version  # mlflow-skinny suffices

    if not uri.startswith("models:/") or "@" not in uri:
        raise ModelLoadError(f"expected models:/<name>@<alias>, got {uri}")
    name, model_alias = uri.removeprefix("models:/").split("@", 1)
    try:
        mv = client(tracking_uri).get_model_version_by_alias(name, model_alias)
    except Exception as e:
        raise ModelLoadError(f"cannot resolve {uri}: {e}") from e
    local = download_model_version(tracking_uri, name, str(mv.version))
    try:
        bundle_dir = find_bundle_dir(local)
    except (FileNotFoundError, ValueError) as e:
        raise ModelLoadError(f"{uri} v{mv.version}: {e}") from e
    slot = _load_bundle_dir(bundle_dir, alias)
    return ModelSlot(alias, str(mv.version), slot.bundle)


class ModelManager:
    """Holds the current slots. ``reload`` builds new slots first, then swaps one reference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.feature_set: FeatureSet = feature_set_for(settings, settings.serving.dataset)
        self._slots: tuple[ModelSlot, ModelSlot | None] | None = None
        self.canary_pct: int = settings.serving.canary_pct  # changed at runtime via /admin/canary

    @property
    def champion(self) -> ModelSlot | None:
        return None if self._slots is None else self._slots[0]

    @property
    def challenger(self) -> ModelSlot | None:
        return None if self._slots is None else self._slots[1]

    def versions(self) -> dict[str, str | None]:
        c, ch = self.champion, self.challenger
        return {"champion": c.version if c else None, "challenger": ch.version if ch else None}

    def _check(self, slot: ModelSlot) -> ModelSlot:
        expected = self.feature_set.spec_hash()
        got = slot.bundle.spec.metadata.get("feature_set_hash")
        if got != expected:
            raise ModelLoadError(
                f"{slot.alias} v{slot.version} was trained on feature set {str(got)[:12]}, "
                f"serving computes {expected[:12]}: refusing to serve (train/serve skew)"
            )
        missing = set(slot.bundle.spec.full.feature_names) - set(self.feature_set.names)
        if missing:
            raise ModelLoadError(f"{slot.alias}: model needs unknown features {sorted(missing)}")
        return slot

    def load_slots(self) -> tuple[ModelSlot, ModelSlot | None]:
        s = self.settings.serving
        if s.model_source == "bundle":
            champion = _load_bundle_dir(s.bundle_dir, "champion")
            challenger = None
            if s.load_challenger and s.challenger_bundle_dir is not None:
                try:  # optional, as with MLflow: a missing challenger must not stop the champion
                    challenger = _load_bundle_dir(s.challenger_bundle_dir, "challenger")
                except ModelLoadError as e:
                    log.warning("no challenger loaded: %s", e)
        else:
            uri = self.settings.training.mlflow_tracking_uri
            champion = _load_mlflow(s.champion_uri, "champion", uri)
            challenger = None
            if s.load_challenger:
                try:
                    challenger = _load_mlflow(s.challenger_uri, "challenger", uri)
                except ModelLoadError as e:
                    log.warning("no challenger loaded: %s", e)
        return self._check(champion), None if challenger is None else self._check(challenger)

    def reload(self) -> dict[str, str | None]:
        slots = self.load_slots()  # may raise; the current slots stay in place
        self._slots = slots  # single reference assignment: atomic for concurrent readers
        log.info("models loaded: %s", self.versions())
        return self.versions()

    def route(self, uid: str) -> ModelSlot | None:
        """Challenger for the canary share of uids (if loaded), else champion."""
        slots = self._slots
        if slots is None:
            return None
        champion, challenger = slots
        pct = self.canary_pct
        if challenger is not None and pct > 0 and canary_bucket(uid) < pct:
            return challenger
        return champion
