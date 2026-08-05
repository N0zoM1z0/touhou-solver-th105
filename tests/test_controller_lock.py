from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.controller_lock import ControllerAlreadyRunning, ControllerLock


class ControllerLockTests(unittest.TestCase):
    def test_rejects_a_second_controller_and_records_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            with ControllerLock(path, command="auto-arcade"):
                # Windows byte-range locks intentionally prevent even a second
                # reader from opening the locked region.
                with self.assertRaises(ControllerAlreadyRunning):
                    with ControllerLock(path, command="fight"):
                        pass
            owner = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(owner["command"], "auto-arcade")
            self.assertGreater(owner["pid"], 0)

    def test_releases_lock_for_next_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            with ControllerLock(path, command="fight"):
                pass
            with ControllerLock(path, command="auto-arcade"):
                pass
            owner = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(owner["command"], "auto-arcade")


if __name__ == "__main__":
    unittest.main()
