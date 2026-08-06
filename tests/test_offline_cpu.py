from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offline_cpu import (
    FEATURE_NAMES,
    distillation_context,
    feature_vector,
    outcome_targets,
    temporal_episode_split,
)


def _record(episode: str, *, action: str = "236B") -> dict[str, object]:
    return {
        "episode_id": episode,
        "difficulty": "lunatic",
        "opponent": "reimu@lunatic",
        "action": action,
        "duration_frames": 31,
        "state": {
            "relative_x_q4": 18,
            "relative_y_q4": 0,
            "closing_speed_q4": -4,
            "self": {
                "character_vtable": "sakuya",
                "x_q4": 10,
                "hp_bp": 7800,
                "spirit_bp": 6500,
                "action": 12,
            },
            "enemy": {
                "character_vtable": "reimu",
                "x_q4": 300,
                "hp_bp": 4200,
                "spirit_bp": 9000,
                "action": 308,
            },
            "enemy_projectiles": [
                {
                    "action": 800,
                    "relative_x_q4": 9,
                    "relative_y_q4": -2,
                    "velocity_x_q4": -8,
                }
            ],
            "own_projectiles": [],
        },
        "outcome": {
            "damage_bp": 900,
            "self_damage_bp": 100,
            "spirit_cost_bp": 2000,
            "punished": True,
            "terminal": "win",
        },
    }


class OfflineCpuTests(unittest.TestCase):
    def test_feature_vector_is_stable_and_includes_nearest_projectile(self) -> None:
        vector = feature_vector(_record("a"))
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        values = dict(zip(FEATURE_NAMES, vector))
        self.assertEqual(values["option"], "236B")
        self.assertEqual(values["enemy_projectile_count"], 1.0)
        self.assertEqual(values["enemy_projectile_nearest_action"], 800.0)

    def test_outcomes_remain_separate_heads(self) -> None:
        targets = outcome_targets(_record("a"))
        self.assertEqual(targets["damage_bp"], 900.0)
        self.assertEqual(targets["connection_probability"], 1.0)
        self.assertEqual(targets["self_damage_bp"], 100.0)
        self.assertEqual(targets["punished_probability"], 1.0)
        self.assertEqual(targets["commitment_frames"], 31.0)
        self.assertEqual(targets["terminal_value"], 1.0)

    def test_distillation_context_is_bounded_and_difficulty_specific(self) -> None:
        context = distillation_context(_record("a"))
        self.assertIn("lunatic", context)
        self.assertIn("close", context)
        self.assertIn("ea308", context)
        self.assertIn("p1-3", context)

    def test_temporal_split_keeps_complete_episodes_together(self) -> None:
        records = [_record("a"), _record("a"), _record("b"), _record("b")]
        train, validation = temporal_episode_split(records, validation_fraction=0.5)
        self.assertEqual(train, [0, 1])
        self.assertEqual(validation, [2, 3])


if __name__ == "__main__":
    unittest.main()
