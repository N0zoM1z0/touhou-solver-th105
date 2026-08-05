"""Bounded, between-encounter compression for TH105 telemetry."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path


class BoundedJsonlWriter:
    """Append cheaply in combat; rotate and gzip only at construction time."""

    def __init__(
        self,
        path: Path,
        *,
        rotate_bytes: int = 8 * 1024 * 1024,
        backups: int = 4,
    ) -> None:
        if rotate_bytes <= 0 or backups <= 0:
            raise ValueError("telemetry limits must be positive")
        self.path = path
        self.rotate_bytes = rotate_bytes
        self.backups = backups
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size >= rotate_bytes:
            self._rotate()

    def _backup(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}.gz")

    def _rotate(self) -> None:
        oldest = self._backup(self.backups)
        if oldest.exists():
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
