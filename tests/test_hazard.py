from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.hazard import HazardProjectile, MovementCandidate, evaluate_paths_reference


class HazardReferenceTests(unittest.TestCase):
    def test_moving_away_beats_standing_still(self) -> None:
        projectiles = (HazardProjectile(100.0, 0.0, -10.0, 0.0, 8.0, 8.0),)
        candidates = (MovementCandidate(0.0, 0.0), MovementCandidate(-8.0, 0.0))
        still, retreat = evaluate_paths_reference(
            0.0, 0.0, 10.0, 20.0, projectiles, candidates, horizon=12
        )
        self.assertFalse(still.safe)
        self.assertTrue(retreat.safe)
        self.assertGreater(retreat.minimum_clearance, still.minimum_clearance)

    def test_empty_world_has_infinite_clearance(self) -> None:
        result = evaluate_paths_reference(
            5.0,
            7.0,
            1.0,
            1.0,
            (),
            (MovementCandidate(2.0, -1.0),),
            horizon=4,
        )[0]
        self.assertTrue(result.safe)
        self.assertTrue(math.isinf(result.minimum_clearance))
        self.assertEqual((result.final_x, result.final_y), (13.0, 3.0))

    def test_graze_window_ignores_early_projectile_overlap(self) -> None:
        projectile = HazardProjectile(20.0, 0.0, 0.0, 0.0, 8.0, 8.0)
        unsafe, graze = evaluate_paths_reference(
            0.0,
            0.0,
            10.0,
            20.0,
            (projectile,),
            (
                MovementCandidate(4.0, 0.0),
                MovementCandidate(4.0, 0.0, graze_frames=5),
            ),
            horizon=5,
        )
        self.assertFalse(unsafe.safe)
        self.assertTrue(graze.safe)

    def test_startup_delay_can_make_a_dash_too_late(self) -> None:
        projectile = HazardProjectile(35.0, 0.0, -8.0, 0.0, 4.0, 4.0)
        immediate, delayed = evaluate_paths_reference(
            0.0,
            0.0,
            5.0,
            10.0,
            (projectile,),
            (
                MovementCandidate(-8.0, 0.0),
                MovementCandidate(-8.0, 0.0, startup_frames=4),
            ),
            horizon=6,
        )
        self.assertTrue(immediate.safe)
        self.assertFalse(delayed.safe)
        self.assertEqual(delayed.final_x, -16.0)

    def test_projectile_acceleration_changes_future_collision(self) -> None:
        linear, accelerating = (
            HazardProjectile(100.0, 0.0, -2.0, 0.0, 4.0, 4.0),
            HazardProjectile(
                100.0, 0.0, -2.0, 0.0, 4.0, 4.0,
                acceleration_x=-2.0,
            ),
        )
        candidate = (MovementCandidate(0.0, 0.0),)
        self.assertTrue(
            evaluate_paths_reference(
                0, 0, 5, 10, (linear,), candidate, horizon=10
            )[0].safe
        )
        self.assertFalse(
            evaluate_paths_reference(
                0, 0, 5, 10, (accelerating,), candidate, horizon=10
            )[0].safe
        )

    def test_pose_center_offset_models_crouching_without_teleporting_startup(self) -> None:
        projectile = HazardProjectile(0.0, 75.0, 0.0, 0.0, 10.0, 10.0)
        standing, crouching = evaluate_paths_reference(
            0.0,
            80.0,
            20.0,
            40.0,
            (projectile,),
            (
                MovementCandidate(0.0, 0.0),
                MovementCandidate(
                    0.0,
                    0.0,
                    half_width=20.0,
                    half_height=20.0,
                    startup_frames=1,
                    center_offset_y=-50.0,
                ),
            ),
            horizon=3,
        )
        self.assertFalse(standing.safe)
        # Startup frame still overlaps, so an already-arrived hit cannot be
        # dodged retroactively merely by selecting the crouch candidate.
        self.assertFalse(crouching.safe)
        self.assertEqual(crouching.final_y, 30.0)

    def test_candidate_acceleration_models_jump_apex_and_descent(self) -> None:
        projectile = HazardProjectile(0.0, 40.0, 0.0, 0.0, 5.0, 5.0)
        linear, ballistic = evaluate_paths_reference(
            0.0,
            0.0,
            5.0,
            5.0,
            (projectile,),
            (
                MovementCandidate(0.0, 10.0),
                MovementCandidate(0.0, 10.0, acceleration_y=-2.0),
            ),
            horizon=10,
        )
        self.assertFalse(linear.safe)
        self.assertTrue(ballistic.safe)
        self.assertEqual(linear.final_y, 100.0)
        self.assertEqual(ballistic.final_y, 0.0)
        self.assertLess(ballistic.final_y, linear.final_y)

    def test_delayed_attack_window_allows_preemptive_escape(self) -> None:
        always = HazardProjectile(0.0, 0.0, 0.0, 0.0, 5.0, 5.0)
        delayed = HazardProjectile(
            0.0,
            0.0,
            0.0,
            0.0,
            5.0,
            5.0,
            active_start_frame=3,
            active_end_frame=5,
        )
        candidate = (MovementCandidate(-10.0, 0.0),)
        self.assertFalse(
            evaluate_paths_reference(0, 0, 5, 5, (always,), candidate, horizon=5)[0].safe
        )
        result = evaluate_paths_reference(
            0, 0, 5, 5, (delayed,), candidate, horizon=5
        )[0]
        self.assertTrue(result.safe)
        self.assertGreater(result.minimum_clearance, 0.0)


if __name__ == "__main__":
    unittest.main()
