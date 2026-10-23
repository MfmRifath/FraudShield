"""Download raw datasets from Kaggle with kagglehub (``data`` dependency group).

* Sparkov (public CC0 dataset) downloads without credentials.
* IEEE-CIS is a competition: it needs Kaggle credentials (``KAGGLE_API_TOKEN``,
  ``~/.kaggle/access_token`` or ``~/.kaggle/kaggle.json``) **and** the competition rules
  accepted at https://www.kaggle.com/competitions/ieee-fraud-detection/rules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fraudshield.config import DatasetConfig

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    pass


def download(cfg: DatasetConfig, target: Path, expected: list[str], force: bool = False) -> None:
    """Download ``cfg`` into ``target`` unless all ``expected`` files are already present."""
    if not force and all((target / name).is_file() for name in expected):
        log.info("raw files already present in %s; skipping download", target)
        return
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover - depends on installed groups
        raise DownloadError("kagglehub missing: run `uv sync --group data`") from e

    target.mkdir(parents=True, exist_ok=True)
    fetch = (
        kagglehub.dataset_download
        if cfg.kaggle_kind == "dataset"
        else kagglehub.competition_download
    )
    log.info("downloading %s %s -> %s", cfg.kaggle_kind, cfg.kaggle_handle, target)
    try:
        fetch(cfg.kaggle_handle, output_dir=str(target), force_download=force)
    except Exception as e:  # kagglehub raises several unrelated types for auth/consent errors
        hint = (
            " Competition data needs Kaggle credentials and accepted rules: "
            f"https://www.kaggle.com/competitions/{cfg.kaggle_handle}/rules"
            if cfg.kaggle_kind == "competition"
            else ""
        )
        raise DownloadError(f"download of {cfg.kaggle_handle} failed: {e}.{hint}") from e
    missing = [n for n in expected if not (target / n).is_file()]
    if missing:
        raise DownloadError(f"download finished but files are missing from {target}: {missing}")
