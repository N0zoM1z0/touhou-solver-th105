from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.motion import (
    build_beginner_ground_combo,
    build_close_normal_chain,
    build_dash_frames,
    build_flight_frames,
    build_jump_frames,
    build_motion_frames,
)


class MotionTests(unittest.TestCase):
    def test_236_mirrors_with_facing(self) -> None:
        right = build_motion_frames("236", "x", "right", recovery_frames=0)
        left = build_motion_frames("236", "x", "left", recovery_frames=0)
        self.assertEqual(right[0], {"down"})
        self.assertEqual(right[2], {"down", "right"})
        self.assertEqual(right[-1], {"right", "x"})
        self.assertEqual(left[2], {"down", "left"})
        self.assertEqual(left[-1], {"left", "x"})

    def test_214_and_623_sequences(self) -> None:
        qcb = build_motion_frames("214", "c", "right", recovery_frames=0)
        dp = build_motion_frames("623", "x", "right", recovery_frames=0)
        self.assertIn({"down", "left"}, qcb)
        self.assertEqual(qcb[-1], {"left", "c"})
        self.assertEqual(dp[0], {"right"})
        self.assertIn({"down", "right", "x"}, dp)

    def test_close_chain_contains_3a_style_input(self) -> None:
        chain = build_close_normal_chain("right")
        self.assertIn({"z"}, chain)
        self.assertIn({"down", "right", "z"}, chain)

    def test_beginner_combo_has_three_a_edges_and_236b_finish(self) -> None:
        combo = build_beginner_ground_combo("right")
        z_edges = sum(
            "z" in chord and (index == 0 or "z" not in combo[index - 1])
            for index, chord in enumerate(combo)
        )
        self.assertEqual(z_edges, 3)
        self.assertIn({"down", "x"}, combo)
        self.assertIn({"c"}, combo)
        self.assertIn({"right", "x"}, combo)

    def test_dash_jump_and_flight_primitives(self) -> None:
        dash = build_dash_frames("right")
        self.assertEqual(dash[:2], [{"right"}, {"right"}])
        self.assertEqual(dash[2], set())
        super_jump = build_jump_frames("right", "back", super_jump=True)
        self.assertEqual(super_jump[0], {"down"})
        self.assertIn({"up", "left"}, super_jump)
        flight = build_flight_frames({"up", "right"})
        self.assertEqual(flight[0], {"a", "up", "right"})


if __name__ == "__main__":
    unittest.main()
