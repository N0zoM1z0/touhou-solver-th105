from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.telemetry import BoundedJsonlWriter


class BoundedTelemetryTests(unittest.TestCase):
    def test_rotation_is_bounded_and_gzipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            path.write_text("x" * 100, encoding="utf-8")
            writer = BoundedJsonlWriter(path, rotate_bytes=50, backups=2)
            backup = Path(f"{path}.1.gz")
            self.assertTrue(backup.is_file())
            with gzip.open(backup, "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "x" * 100)
            writer.write({"event": "heartbeat", "frame": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["frame"], 1
            )

    def test_oldest_backup_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            for generation in range(3):
                path.write_text(str(generation) * 20, encoding="utf-8")
                BoundedJsonlWriter(path, rotate_bytes=10, backups=2)
            self.assertTrue(Path(f"{path}.1.gz").is_file())
            self.assertTrue(Path(f"{path}.2.gz").is_file())
            self.assertFalse(Path(f"{path}.3.gz").exists())


if __name__ == "__main__":
    unittest.main()
