"""Data manifests: what raw files a build used, and a content hash of what it produced.

Manifests are small JSON files under ``data/manifests/`` and are committed, so anyone can check
that their raw download and rebuilt feature table match the reference build.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict

_CHUNK = 1 << 20


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def frame_content_sha256(df: pl.DataFrame, batch_rows: int = 200_000) -> str:
    """Hash of a frame's *logical* content: schema + a canonical CSV rendering of every row.

    Independent of chunking and Arrow buffer layout (raw IPC bytes are not: validity bitmaps,
    string-view buffers and values under null slots vary). Floats use shortest round-trip
    formatting, nulls are ``\\N`` and non-numeric values are quoted, so "" and null differ.
    """
    h = hashlib.sha256(sha256_json([[n, str(t)] for n, t in df.schema.items()]).encode())
    for offset in range(0, df.height, batch_rows):
        buf = io.BytesIO()
        df.slice(offset, batch_rows).write_csv(
            buf, include_header=False, null_value="\\N", quote_style="non_numeric"
        )
        h.update(buf.getvalue())
    return h.hexdigest()


class RawFile(_Model):
    name: str
    bytes: int
    sha256: str
    rows: int


class RawManifest(_Model):
    dataset: str
    source: str
    license: str
    files: tuple[RawFile, ...]

    def content_hash(self) -> str:
        """Stable identifier of the raw inputs (logged with every model in Phase 2)."""
        return sha256_json([[f.name, f.sha256] for f in self.files])


class FeaturesManifest(_Model):
    dataset: str
    raw_manifest_hash: str
    feature_set_hash: str
    label_delay: dict[str, Any]
    feature_names: tuple[str, ...]
    rows: int
    canonical_sha256: str
    features_sha256: str
    versions: dict[str, str]


def csv_rows(path: Path) -> int:
    return int(pl.scan_csv(path, infer_schema=False).select(pl.len()).collect().item())


def record_raw(dataset: str, source: str, license_: str, files: list[Path]) -> RawManifest:
    entries = tuple(
        RawFile(name=f.name, bytes=f.stat().st_size, sha256=sha256_file(f), rows=csv_rows(f))
        for f in files
    )
    return RawManifest(dataset=dataset, source=source, license=license_, files=entries)


def verify_raw(manifest: RawManifest, raw_dir: Path) -> list[str]:
    """Return human-readable problems (empty list = files match the manifest)."""
    problems = []
    for entry in manifest.files:
        path = raw_dir / entry.name
        if not path.is_file():
            problems.append(f"{entry.name}: missing")
        elif path.stat().st_size != entry.bytes:
            problems.append(f"{entry.name}: size {path.stat().st_size} != {entry.bytes}")
        elif sha256_file(path) != entry.sha256:
            problems.append(f"{entry.name}: sha256 mismatch")
    return problems


def write_json(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n")


def read_json[M: BaseModel](cls: type[M], path: Path) -> M:
    return cls.model_validate_json(path.read_text())
