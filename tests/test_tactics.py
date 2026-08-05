from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.tactics import ActionCandidate, TacticalContext, rank_actions, score_action


class TacticalScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.direct = ActionCandidate("236B", 14, 70, 1200, 200, 250, 900)
        self.setup = ActionCandidate("214B", 18, 48, 850, 200, 280, 800, "setup")

    def test_rejects_attack_that_cannot_land_in_recovery(self) -> None:
        result = score_action(
            self.direct, TacticalContext(400, 0, 1000, "recovery", 10, 0, False)
        )
        self.assertEqual(result.reason, "late-punish")

    def test_empty_layout_can_prefer_setup(self) -> None:
        ranked = rank_actions(
            (self.direct, self.setup),
            TacticalContext(400, 0, 1000, "neutral", 0, 0, True),
        )
        self.assertEqual(ranked[0].candidate.name, "214B")

    def test_existing_knives_reduce_duplicate_setup_value(self) -> None:
        ranked = rank_actions(
            (self.direct, self.setup),
            TacticalContext(400, 0, 1000, "neutral", 0, 18, False),
        )
        self.assertEqual(ranked[0].candidate.name, "236B")

    def test_spell_disallows_offense(self) -> None:
        result = score_action(
            self.direct,
            TacticalContext(400, 0, 1000, "spell-danger", 0, 0, False),
        )
        self.assertEqual(result.reason, "enemy-spell")

    def test_closing_enemy_can_intercept_slow_setup_startup(self) -> None:
        result = score_action(
            self.setup,
            TacticalContext(
                300, 0, 1000, "neutral", 0, 0, True,
                enemy_time_to_contact=17,
            ),
        )
        self.assertEqual(result.reason, "startup-intercept")

    def test_observed_punishes_lower_future_score(self) -> None:
        risky = ActionCandidate(
            "risky", 14, 70, 1200, 200, 250, 900, punish_rate=0.5
        )
        context = TacticalContext(400, 0, 1000, "neutral", 0, 8, False)
        self.assertLess(score_action(risky, context).score, score_action(self.direct, context).score)


if __name__ == "__main__":
    unittest.main()
