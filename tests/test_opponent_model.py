from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.opponent_model import OpponentActionModel


class OpponentActionModelTests(unittest.TestCase):
    def test_learns_active_and_recovery_windows(self) -> None:
        model = OpponentActionModel()
        model.observe(0, enemy_action=0, me_hp=10000, projectile_count=0)
        model.observe(10, enemy_action=400, me_hp=10000, projectile_count=0)
        active = model.observe(18, enemy_action=400, me_hp=10000, projectile_count=1)
        self.assertEqual(active.phase, "active")
        recovery = model.observe(24, enemy_action=400, me_hp=10000, projectile_count=1)
        self.assertEqual(recovery.phase, "recovery")
        model.observe(40, enemy_action=0, me_hp=10000, projectile_count=1)
        profile = model.profiles[400]
        self.assertEqual(profile.completions, 1)
        self.assertEqual(profile.total_frames, 30.0)

    def test_spell_is_never_early_punish_window(self) -> None:
        model = OpponentActionModel()
        assessment = model.observe(
            0, enemy_action=600, me_hp=10000, projectile_count=0
        )
        self.assertTrue(assessment.spell)
        self.assertEqual(assessment.phase, "spell-danger")
        self.assertEqual(assessment.punish_window, 0.0)

    def test_native_attack_box_marks_active_without_sacrificing_hp(self) -> None:
        model = OpponentActionModel()
        model.observe(0, enemy_action=0, me_hp=10000, projectile_count=0)
        model.observe(10, enemy_action=408, me_hp=10000, projectile_count=0)
        active = model.observe(
            15,
            enemy_action=408,
            me_hp=10000,
            projectile_count=0,
            active_hitbox=True,
        )
        self.assertEqual(active.phase, "active")
        profile = model.profiles[408]
        self.assertEqual(profile.first_active_box_frame, 5.0)
        self.assertEqual(profile.last_active_box_frame, 5.0)

    def test_hit_reaction_is_exposed_without_offensive_episode(self) -> None:
        model = OpponentActionModel()
        assessment = model.observe(
            5, enemy_action=71, me_hp=10000, projectile_count=0
        )
        self.assertEqual(assessment.phase, "reaction")


if __name__ == "__main__":
    unittest.main()
