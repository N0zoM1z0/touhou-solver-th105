from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.difficulty import DifficultyCurriculum


class DifficultyCurriculumTests(unittest.TestCase):
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
