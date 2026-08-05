from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.input import KEYS, _canonical_names


class InputMappingTests(unittest.TestCase):
    def test_profile_1p_default_scancodes(self) -> None:
        self.assertEqual(KEYS["up"].scan_code, 0x48)
        self.assertEqual(KEYS["down"].scan_code, 0x50)
        self.assertEqual(KEYS["left"].scan_code, 0x4B)
        self.assertEqual(KEYS["right"].scan_code, 0x4D)
        self.assertEqual([KEYS[k].dik_code for k in ("up", "down", "left", "right")], [0xC8, 0xD0, 0xCB, 0xCD])
        self.assertEqual([KEYS[k].scan_code for k in "zxc"], [0x2C, 0x2D, 0x2E])
        self.assertEqual([KEYS[k].scan_code for k in "asd"], [0x1E, 0x1F, 0x20])

    def test_aliases_release_once(self) -> None:
        names = _canonical_names(set(KEYS.values()))
        self.assertIn("z", names)
        self.assertNotIn("confirm", names)
        self.assertIn("x", names)
        self.assertNotIn("cancel", names)
        self.assertEqual(len(names), len(set(KEYS.values())))


if __name__ == "__main__":
    unittest.main()
