from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.sakuya_combos import ComboContext, select_combo


class SakuyaComboSelectionTests(unittest.TestCase):
    def test_full_spirit_selects_guide_beginner_route(self) -> None:
        combo = select_combo(ComboContext(55.0, 640.0, 0.0, 1000))
        self.assertIsNotNone(combo)
        self.assertEqual(combo.name, "AAA-2B-C-236B")

    def test_low_spirit_preserves_reserve_with_meterless_route(self) -> None:
        combo = select_combo(ComboContext(55.0, 640.0, 0.0, 350))
        self.assertIsNotNone(combo)
        self.assertEqual(combo.name, "AAAA-meterless")

    def test_corner_enables_one_orb_route(self) -> None:
        combo = select_combo(ComboContext(55.0, 1200.0, 0.0, 500))
        self.assertIsNotNone(combo)
        self.assertEqual(combo.name, "AAAA-623B")

    def test_ground_routes_do_not_chase_airborne_enemy(self) -> None:
        self.assertIsNone(select_combo(ComboContext(70.0, 640.0, 100.0, 1000)))

    def test_mid_range_uses_two_a_instead_of_far_a_for_aaa(self) -> None:
        combo = select_combo(ComboContext(80.0, 640.0, 0.0, 1000))
        self.assertIsNotNone(combo)
        self.assertEqual(combo.name, "2A-2B-C-236B")


if __name__ == "__main__":
    unittest.main()
