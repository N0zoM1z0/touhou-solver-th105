from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.motion import (
    build_dash_frames,
    build_flight_frames,
    build_jump_frames,
    build_motion_frames,
    generated_attack_chord_hypotheses,
    generated_attack_probe_hypotheses,
    generated_motion_hypotheses,
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

    def test_dash_jump_and_flight_primitives(self) -> None:
        dash = build_dash_frames("right")
        self.assertEqual(dash[:2], [{"right"}, {"right"}])
        self.assertEqual(dash[2], set())
        super_jump = build_jump_frames("right", "back", super_jump=True)
        self.assertEqual(super_jump[0], {"down"})
        self.assertIn({"up", "left"}, super_jump)
        flight = build_flight_frames({"up", "right"})
        self.assertEqual(flight[0], {"a", "up", "right"})

    def test_generated_catalog_contains_unique_double_down_hypotheses(self) -> None:
        catalog = generated_motion_hypotheses()
        labels = [label for label, _motion, _button in catalog]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(labels), 15)
        self.assertIn(("22B", "22", "x"), catalog)
        self.assertIn(("22C", "22", "c"), catalog)

    def test_attack_probe_catalog_is_cartesian_not_character_specific(self) -> None:
        catalog = generated_attack_probe_hypotheses()
        labels = [label for label, _pattern in catalog]
        self.assertEqual(len(labels), 54)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn(
            ("down+toward+x", "toward@3,down+toward+x@2,neutral@4"),
            catalog,
        )
        self.assertIn(
            ("a+up+back+c", "toward@3,a+up+back+c@2,neutral@4"),
            catalog,
        )
        self.assertIn(
            ("down+toward+x", frozenset({"down", "toward", "x"})),
            generated_attack_chord_hypotheses(),
        )


if __name__ == "__main__":
    unittest.main()
