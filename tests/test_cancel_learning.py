from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.cancel_learning import CancelGraphModel


class CancelGraphTests(unittest.TestCase):
    def test_records_only_attack_grounded_native_transition(self) -> None:
        model = CancelGraphModel()
        model.observe_transition(
            source_action=300,
            source_sequence=0,
            source_pose=3,
            source_frame=2,
            target_action=305,
            target_sequence=0,
            normalized_keys=frozenset({"toward", "z"}),
            direct_damage=120,
        )
        model.observe_transition(
            source_action=300,
            source_sequence=0,
            source_pose=3,
            source_frame=2,
            target_action=0,
            target_sequence=0,
            normalized_keys=frozenset({"z"}),
        )
        metrics = model.metrics()
        self.assertEqual(
            (metrics["edges"], metrics["trials"], metrics["direct_damage_events"]),
            (1, 1, 1),
        )

    def test_candidate_requires_observed_pose_and_timing_window(self) -> None:
        model = CancelGraphModel()
        model.observe_transition(
            source_action=302,
            source_sequence=0,
            source_pose=4,
            source_frame=6,
            target_action=418,
            target_sequence=0,
            normalized_keys=frozenset({"a", "toward", "c"}),
        )
        self.assertEqual(
            len(model.candidates(action_id=302, sequence=0, pose=4, action_frame=7)),
            1,
        )
        self.assertFalse(
            model.candidates(action_id=302, sequence=0, pose=3, action_frame=7)
        )
        self.assertFalse(
            model.candidates(action_id=302, sequence=0, pose=4, action_frame=12)
        )

    def test_round_trip(self) -> None:
        first = CancelGraphModel()
        first.observe_transition(
            source_action=300,
            source_sequence=0,
            source_pose=1,
            source_frame=3,
            target_action=301,
            target_sequence=0,
            normalized_keys=frozenset({"z"}),
        )
        second = CancelGraphModel()
        second.import_state(first.export_state())
        self.assertEqual(second.export_state(), first.export_state())

    def test_rising_edge_binds_delayed_native_transition_once(self) -> None:
        model = CancelGraphModel()
        model.record_input(
            frame=10,
            source_action=300,
            source_sequence=0,
            source_pose=2,
            source_frame=4,
            normalized_keys=frozenset({"toward", "z"}),
        )
        self.assertFalse(model.observe_actor(frame=12, action_id=300, sequence=0))
        self.assertTrue(
            model.observe_actor(
                frame=14,
                action_id=305,
                sequence=0,
                direct_damage=100,
            )
        )
        metrics = model.metrics()
        self.assertEqual(metrics["input_edges"], 1)
        self.assertEqual(metrics["accepted_transitions"], 1)
        self.assertEqual(metrics["edges"], 1)
        # Held Z is not a second causal edge.
        model.record_input(
            frame=15,
            source_action=305,
            source_sequence=0,
            source_pose=0,
            source_frame=0,
            normalized_keys=frozenset({"toward", "z"}),
        )
        self.assertEqual(model.metrics()["input_edges"], 1)

    def test_expired_input_does_not_create_false_cancel(self) -> None:
        model = CancelGraphModel()
        model.record_input(
            frame=0,
            source_action=300,
            source_sequence=0,
            source_pose=1,
            source_frame=2,
            normalized_keys=frozenset({"x"}),
        )
        model.observe_actor(frame=7, action_id=300, sequence=0)
        self.assertEqual(model.metrics()["expired_inputs"], 1)
        self.assertFalse(model.observe_actor(frame=8, action_id=400, sequence=0))
        self.assertEqual(model.metrics()["edges"], 0)


if __name__ == "__main__":
    unittest.main()
