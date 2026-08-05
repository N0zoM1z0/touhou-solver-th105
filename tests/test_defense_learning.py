from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.defense_learning import DefenseResponseModel


class DefenseResponseTests(unittest.TestCase):
    def test_unknown_uses_conservative_first_response(self) -> None:
        model = DefenseResponseModel()
        self.assertEqual(
            model.choose("300:0", ("high_guard", "low_guard")), "high_guard"
        )

    def test_damage_causes_one_alternative_trial(self) -> None:
        model = DefenseResponseModel()
        model.begin("302:0", "high_guard", hp=10000, spirit=1000, distance=80)
        model.mark_applied()
        model.observe(hp=9500, spirit=1000, distance=70)
        model.finish()
        self.assertEqual(
            model.choose("302:0", ("high_guard", "low_guard")), "low_guard"
        )

    def test_successful_alternative_converges(self) -> None:
        model = DefenseResponseModel()
        model.begin("302:0", "high_guard", hp=10000, spirit=1000, distance=80)
        model.mark_applied()
        model.observe(hp=9500, spirit=1000, distance=70)
        model.finish()
        model.begin("302:0", "low_guard", hp=10000, spirit=1000, distance=80)
        model.mark_applied()
        model.observe(hp=10000, spirit=930, distance=65)
        model.finish()
        self.assertEqual(
            model.choose("302:0", ("high_guard", "low_guard")), "low_guard"
        )

    def test_whiff_does_not_create_false_success(self) -> None:
        model = DefenseResponseModel()
        model.begin("300:0", "high_guard", hp=10000, spirit=1000, distance=400)
        model.mark_applied()
        model.finish()
        self.assertNotIn("300:0", model.table)

    def test_round_trip(self) -> None:
        first = DefenseResponseModel()
        first.begin("304:0", "high_guard", hp=10000, spirit=1000, distance=60)
        first.mark_applied()
        first.observe(hp=10000, spirit=900, distance=50)
        first.finish()
        second = DefenseResponseModel()
        second.import_state(first.export_state())
        stats = second.table["304:0"]["high_guard"]
        self.assertEqual(
            (stats.trials, stats.successes, stats.total_guard_cost),
            (1, 1, 100),
        )

    def test_unapplied_response_does_not_poison_learning(self) -> None:
        model = DefenseResponseModel()
        model.begin("305:0", "high_guard", hp=10000, spirit=1000, distance=70)
        model.observe(hp=9000, spirit=1000, distance=60)
        model.finish()
        self.assertNotIn("305:0", model.table)


if __name__ == "__main__":
    unittest.main()
