from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.evaluation import evaluate_knowledge, wilson_interval


class LearningEvaluationTests(unittest.TestCase):
    def test_uses_global_offense_row_without_double_counting_contexts(self) -> None:
        stats = {
            "trials": 4,
            "connections": 3,
            "punished_trials": 1,
            "normalized_samples": 4,
            "total_damage_bp": 1200,
            "total_self_damage_bp": 200,
            "total_commitment": 80,
            "self_damage_histogram": [3, 1, 0, 0, 0, 0, 0, 0],
        }
        report = evaluate_knowledge(
            {
                "schema_version": 1,
                "characters": {
                    "aya@hard": {
                        "offense_outcomes": {
                            "*": {"236B": stats},
                            "close:ground:field:neutral": {"236B": stats},
                            "__reward__": {"rounds": {"rounds": 2, "wins": 1}},
                        },
                        "defense_responses": {},
                    }
                },
            }
        )
        row = report["characters"]["aya@hard"]
        self.assertEqual(row["offense"]["trials"], 4)
        self.assertEqual(row["offense"]["skills_observed"], 1)
        self.assertEqual(row["offense"]["connection_rate"], 0.75)
        self.assertEqual(row["rounds"]["win_rate"], 0.5)

    def test_wilson_interval_is_bounded_and_shrinks_with_evidence(self) -> None:
        small = wilson_interval(1, 2)
        large = wilson_interval(50, 100)
        self.assertGreaterEqual(small[0], 0.0)
        self.assertLessEqual(small[1], 1.0)
        self.assertLess(large[1] - large[0], small[1] - small[0])


if __name__ == "__main__":
    unittest.main()
