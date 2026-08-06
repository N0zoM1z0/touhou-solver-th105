from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.difficulty import (
    DifficultyCurriculum,
    FixedRoundDifficultyCycle,
    next_cyclic_difficulty,
)


class DifficultyCurriculumTests(unittest.TestCase):
    def test_fixed_cycle_waits_for_exact_round_quota(self) -> None:
        cycle = FixedRoundDifficultyCycle(2, rounds_per_difficulty=6)
        cycle.record(wins=3, losses=2)
        self.assertEqual(cycle.choose(), "hard")
        self.assertEqual(cycle.remaining_rounds, 1)
        cycle.record(draws=1)
        self.assertTrue(cycle.rotation_due)
        self.assertEqual(cycle.choose(), "lunatic")
        self.assertEqual(cycle.completed_rounds, 0)

    def test_fixed_cycle_wraps_and_rejects_bad_quota(self) -> None:
        cycle = FixedRoundDifficultyCycle(3, rounds_per_difficulty=1)
        cycle.record(losses=1)
        self.assertEqual(cycle.choose(), "easy")
        with self.assertRaises(ValueError):
            FixedRoundDifficultyCycle(0, rounds_per_difficulty=0)

    def test_cycle_advances_and_wraps(self) -> None:
        self.assertEqual(next_cyclic_difficulty(2), "lunatic")
        self.assertEqual(next_cyclic_difficulty(3), "easy")

    def test_cycle_rejects_invalid_level(self) -> None:
        with self.assertRaises(ValueError):
            next_cyclic_difficulty(4)

    def test_promotes_one_level_after_sustained_wins(self) -> None:
        curriculum = DifficultyCurriculum(0)
        curriculum.record(wins=4, losses=2)
        self.assertEqual(curriculum.choose(), "normal")
        self.assertEqual(curriculum.status()["window_rounds"], 0)

    def test_demotes_one_level_after_losses(self) -> None:
        curriculum = DifficultyCurriculum(2)
        curriculum.record(wins=1, losses=5)
        self.assertEqual(curriculum.choose(), "normal")

    def test_requires_fresh_evidence_and_respects_bounds(self) -> None:
        curriculum = DifficultyCurriculum(3)
        curriculum.record(wins=6)
        self.assertEqual(curriculum.choose(), "lunatic")
        # No promotion was possible, so the same window remains observable.
        self.assertEqual(curriculum.status()["window_rounds"], 6)


if __name__ == "__main__":
    unittest.main()
