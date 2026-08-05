from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.attack_geometry import AttackGeometryModel


class AttackGeometryTests(unittest.TestCase):
    def test_learns_pose_envelope_and_active_timing(self) -> None:
        model = AttackGeometryModel()
        model.observe(10, action_id=305, sequence=0, pose=0, attack_boxes=())
        model.observe(
            13,
            action_id=305,
            sequence=0,
            pose=1,
            attack_boxes=((-20, -80, 70, -10), (60, -50, 110, 0)),
        )
        model.observe(
            15,
            action_id=305,
            sequence=0,
            pose=2,
            attack_boxes=((-10, -100, 90, -20),),
        )
        forecast = model.forecast(305, 0)
        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(forecast.first_active_elapsed, 3)
        self.assertEqual(forecast.last_active_elapsed, 5)
        self.assertEqual(forecast.attack_envelope, (-20, -100, 110, 0))
        self.assertEqual(model.metrics()["poses"], 3)

    def test_sequences_do_not_share_geometry(self) -> None:
        model = AttackGeometryModel()
        model.observe(
            0, action_id=400, sequence=0, pose=0, attack_boxes=((0, -10, 20, 0),)
        )
        model.observe(
            1, action_id=400, sequence=1, pose=0, attack_boxes=((0, -10, 80, 0),)
        )
        self.assertEqual(model.forecast(400, 0).attack_envelope[2], 20)  # type: ignore[union-attr]
        self.assertEqual(model.forecast(400, 1).attack_envelope[2], 80)  # type: ignore[union-attr]

    def test_round_trip_preserves_bounded_pose_rows(self) -> None:
        first = AttackGeometryModel()
        first.observe(
            5, action_id=418, sequence=0, pose=7, attack_boxes=((-5, -40, 95, 0),)
        )
        second = AttackGeometryModel()
        second.import_state(first.export_state())
        self.assertEqual(second.export_state(), first.export_state())


if __name__ == "__main__":
    unittest.main()
