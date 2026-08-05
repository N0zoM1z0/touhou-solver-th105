from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.policies.adaptive import _hazard_projectiles, _native_guard_response


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


if __name__ == "__main__":
    unittest.main()
