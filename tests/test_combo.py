from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.combo import ComboStep, parse_combo


class ComboTests(unittest.TestCase):
    def test_parse_chords_and_neutral(self) -> None:
        self.assertEqual(
            parse_combo("right@6,right+z@3,neutral@8"),
            (
                ComboStep(frozenset({"right"}), 6),
                ComboStep(frozenset({"right", "z"}), 3),
                ComboStep(frozenset(), 8),
            ),
        )

    def test_reject_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            parse_combo("fire@3")


if __name__ == "__main__":
    unittest.main()
