from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.policies.adaptive import (
    _demonstration_frames,
    _hazard_projectiles,
    _human_demonstration_utility,
    _pattern_has_advancing_attack,
    _native_guard_response,
)


class AdaptiveNativeGeometryTests(unittest.TestCase):
    def test_frame_attack_box_replaces_learned_square_extent(self) -> None:
        projectile = SimpleNamespace(
            x=100.0,
            y=50.0,
            velocity_x=-4.0,
            velocity_y=2.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            facing=-1,
            attack_boxes=((-30, -10, 10, 20),),
        )
        hazard = _hazard_projectiles(projectile, half_extent=99.0)[0]
        self.assertEqual((hazard.x, hazard.y), (110.0, 45.0))
        self.assertEqual((hazard.half_width, hazard.half_height), (20.0, 15.0))

    def test_empty_native_attack_boxes_mean_inactive_frame(self) -> None:
        projectile = SimpleNamespace(
            x=0.0,
            y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            attack_boxes=(),
        )
        self.assertEqual(_hazard_projectiles(projectile, half_extent=99.0), ())

    def test_unambiguous_native_guard_level_overrides_learning(self) -> None:
        self.assertEqual(_native_guard_response(0x2, "low_guard"), "high_guard")
        self.assertEqual(_native_guard_response(0x4, "high_guard"), "low_guard")
        self.assertEqual(_native_guard_response(0x6, "high_guard"), "high_guard")

    def test_human_pattern_replays_relative_to_current_facing(self) -> None:
        frames = _demonstration_frames(
            "down@1,down+toward@2,toward+x@1,neutral@2", "left"
        )
        self.assertEqual(frames[0], {"down"})
        self.assertEqual(frames[1], {"down", "left"})
        self.assertEqual(frames[3], {"left", "x"})
        self.assertEqual(frames[-1], set())

    def test_human_pattern_rejects_unbounded_or_unknown_input(self) -> None:
        self.assertEqual(_demonstration_frames("toward+q@2", "right"), [])
        self.assertEqual(_demonstration_frames("z@181", "right"), [])

    def test_advancing_attack_accepts_chord_or_short_sequence(self) -> None:
        self.assertTrue(_pattern_has_advancing_attack("a+toward+x@3"))
        self.assertTrue(_pattern_has_advancing_attack("toward@3,up+z@2"))
        self.assertFalse(_pattern_has_advancing_attack("back@3,x@2"))
        self.assertFalse(_pattern_has_advancing_attack("toward@8,z@2"))

    def test_losing_human_trade_is_not_a_replay_candidate(self) -> None:
        losing = {
            "trials": 1,
            "connections": 1,
            "total_damage": 480,
            "total_self_damage": 1125,
            "total_spirit_cost": 588,
            "total_duration": 70,
        }
        self.assertIsNone(
            _human_demonstration_utility(
                losing, spirit=1000, support=1, context_penalty=0.0
            )
        )

    def test_safe_human_hit_has_positive_prior_utility(self) -> None:
        safe = {
            "trials": 1,
            "connections": 1,
            "total_damage": 690,
            "total_self_damage": 0,
            "total_spirit_cost": 200,
            "total_duration": 52,
        }
        utility = _human_demonstration_utility(
            safe, spirit=1000, support=1, context_penalty=0.0
        )
        self.assertIsNotNone(utility)
        assert utility is not None
        self.assertGreater(utility, 0.0)


if __name__ == "__main__":
    unittest.main()
