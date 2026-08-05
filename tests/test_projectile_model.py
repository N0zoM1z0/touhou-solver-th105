from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.projectile_model import ProjectileEnvelopeModel


def projectile(pointer: int, action: int, x: float, y: float, vx: float = 0.0):
    return SimpleNamespace(
        pointer=pointer,
        action_id=action,
        x=x,
        y=y,
        velocity_x=vx,
        velocity_y=0.0,
        acceleration_x=0.0,
        acceleration_y=0.0,
    )


class ProjectileEnvelopeTests(unittest.TestCase):
    def test_disappeared_projectile_is_projected_and_learned(self) -> None:
        model = ProjectileEnvelopeModel()
        model.observe(
            (projectile(0x1000, 812, 60.0, 0.0, -10.0),),
            player_x=0.0,
            player_y=0.0,
            player_half_width=18.0,
            player_half_height=42.0,
            took_damage=False,
            first_contact=True,
        )
        impact = model.observe(
            (),
            player_x=0.0,
            player_y=0.0,
            player_half_width=18.0,
            player_half_height=42.0,
            took_damage=True,
            first_contact=True,
        )
        self.assertIsNotNone(impact)
        assert impact is not None
        self.assertTrue(impact.disappeared)
        self.assertEqual(impact.action_id, 812)
        self.assertEqual(model.extent_for(812), 40.0)

    def test_hitstun_continuation_does_not_learn(self) -> None:
        model = ProjectileEnvelopeModel()
        impact = model.observe(
            (projectile(1, 900, 25.0, 0.0),),
            player_x=0.0,
            player_y=0.0,
            player_half_width=18.0,
            player_half_height=42.0,
            took_damage=True,
            first_contact=False,
        )
        self.assertIsNone(impact)
        self.assertEqual(model.extents, {})

    def test_recently_disappeared_projectile_survives_three_frame_window(self) -> None:
        model = ProjectileEnvelopeModel()
        model.observe(
            (projectile(0x2000, 826, 70.0, 0.0, -10.0),),
            player_x=0.0, player_y=0.0,
            player_half_width=18.0, player_half_height=42.0,
            took_damage=False, first_contact=True,
        )
        for _ in range(2):
            model.observe(
                (), player_x=0.0, player_y=0.0,
                player_half_width=18.0, player_half_height=42.0,
                took_damage=False, first_contact=True,
            )
        impact = model.observe(
            (), player_x=0.0, player_y=0.0,
            player_half_width=18.0, player_half_height=42.0,
            took_damage=True, first_contact=True,
        )
        self.assertIsNotNone(impact)
        assert impact is not None
        self.assertEqual(impact.action_id, 826)

    def test_far_projectile_is_not_misattributed_to_melee(self) -> None:
        model = ProjectileEnvelopeModel()
        impact = model.observe(
            (projectile(1, 900, 300.0, 200.0),),
            player_x=0.0,
            player_y=0.0,
            player_half_width=18.0,
            player_half_height=42.0,
            took_damage=True,
            first_contact=True,
        )
        self.assertIsNone(impact)

    def test_state_round_trip_preserves_calibration(self) -> None:
        first = ProjectileEnvelopeModel()
        first.extents[811] = 55.0
        first.samples[811] = 3
        second = ProjectileEnvelopeModel()
        second.import_state(first.export_state())
        self.assertEqual(second.extent_for(811), 55.0)
        self.assertEqual(second.samples[811], 3)

    def test_equal_distance_candidates_use_stable_pointer_tiebreak(self) -> None:
        model = ProjectileEnvelopeModel()
        impact = model.observe(
            (
                projectile(0x2000, 812, 40.0, 0.0),
                projectile(0x1000, 811, 40.0, 0.0),
            ),
            player_x=0.0,
            player_y=0.0,
            player_half_width=18.0,
            player_half_height=42.0,
            took_damage=True,
            first_contact=True,
        )
        self.assertIsNotNone(impact)
        assert impact is not None
        self.assertEqual((impact.pointer, impact.action_id), (0x1000, 811))


if __name__ == "__main__":
    unittest.main()
