"""Bounded, between-encounter compression for TH105 telemetry."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path


def _gzip_content_sha256(path: Path) -> str:
    """Hash decompressed content so gzip metadata cannot create duplicates."""
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_gzip_shard(
    source: Path,
    archive_dir: Path,
    *,
    logical_name: str,
) -> Path:
    """Atomically preserve one immutable gzip shard under a content hash."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    temporary = archive_dir / f".{logical_name}.{uuid.uuid4().hex}.tmp"
    shutil.copy2(source, temporary)
    try:
        digest = _gzip_content_sha256(temporary)
        destination = archive_dir / f"{logical_name}.{digest}.gz"
        if destination.exists():
            if _gzip_content_sha256(destination) != digest:
                raise RuntimeError(f"corrupt corpus archive shard {destination}")
            return destination
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


class BoundedJsonlWriter:
    """Append cheaply in combat; rotate and gzip only at construction time."""

    def __init__(
        self,
        path: Path,
        *,
        rotate_bytes: int = 8 * 1024 * 1024,
        backups: int = 4,
        archive_dir: Path | None = None,
    ) -> None:
        if rotate_bytes <= 0 or backups <= 0:
            raise ValueError("telemetry limits must be positive")
        self.path = path
        self.rotate_bytes = rotate_bytes
        self.backups = backups
        self.archive_dir = archive_dir
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size >= rotate_bytes:
            self._rotate()

    def _backup(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}.gz")

    def _rotate(self) -> None:
        oldest = self._backup(self.backups)
        if oldest.exists():
            if self.archive_dir is not None:
                archive_gzip_shard(
                    oldest,
                    self.archive_dir,
                    logical_name=self.path.name,
                )
            oldest.unlink()
        for index in range(self.backups, 1, -1):
            previous = self._backup(index - 1)
            if previous.exists():
                os.replace(previous, self._backup(index))
        temporary = self._backup(1).with_suffix(".gz.tmp")
        with self.path.open("rb") as source, gzip.open(temporary, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary, self._backup(1))
        self.path.unlink()

    def write(self, record: dict[str, object]) -> None:
        self.write_many((record,))

    def write_many(self, records: Iterable[dict[str, object]]) -> None:
        rows = tuple(records) if not isinstance(records, tuple) else records
        if not rows:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            for record in rows:
                stream.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                stream.write("\n")
