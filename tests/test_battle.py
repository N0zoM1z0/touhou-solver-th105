from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.battle import BOOTSTRAP_CYCLE, FIGHTER_ACTION_ID, FIGHTER_HP, FIGHTER_POSITION_X


class BootstrapBattleTests(unittest.TestCase):
    def test_cycle_exercises_movement_guard_and_attacks(self) -> None:
        chords = {frozenset(step.keys) for step in BOOTSTRAP_CYCLE}
        self.assertIn(frozenset({"right"}), chords)
        self.assertIn(frozenset({"left"}), chords)
        self.assertIn(frozenset({"down", "left"}), chords)
        self.assertIn(frozenset({"z"}), chords)
        self.assertIn(frozenset({"x"}), chords)

    def test_observed_fighter_offsets(self) -> None:
        self.assertEqual(FIGHTER_POSITION_X, 0xEC)
        self.assertEqual(FIGHTER_ACTION_ID, 0x13C)
        self.assertEqual(FIGHTER_HP, 0x174)


if __name__ == "__main__":
    unittest.main()
