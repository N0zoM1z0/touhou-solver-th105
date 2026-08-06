from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.learning_curve import learning_curve_point


class LearningCurveTests(unittest.TestCase):
    def test_round_point_keeps_compact_autonomous_progress(self) -> None:
        point = learning_curve_point(
            session_id="session",
            terminal_sequence=3,
            opponent="yukari@lunatic",
            difficulty="lunatic",
            playstyle="aggressive",
            won=True,
            me_hp=6000,
            enemy_hp=0,
            max_me_hp=10000,
            max_enemy_hp=10000,
            policy_sha256="a" * 64,
            offline_policy_sha256=None,
            plugin_metrics={
                "counts": {
                    "grammar_probe_attempts:z:z": 4,
                    "cancel_hypothesis_attempts:source:motion:236B": 3,
                    "learned_cancel_attempts:edge": 2,
                },
                "cancel_graph": {"edges": 5, "accepted_transitions": 3},
                "coverage_explorer": {
                    "coverage": 0.25,
                    "covered_scope_actions": 10,
                    "available_scope_actions": 40,
                },
                "offense_outcome_state": {
                    "*": {
                        "grammar-probe:z": {
                            "trials": 4,
                            "connections": 2,
                            "punished_trials": 1,
                            "total_damage_bp": 1200,
                            "total_self_damage_bp": 400,
                        }
                    },
                    "__reward__": {
                        "rounds": {"rounds": 3, "wins": 2, "losses": 1}
                    },
                },
            },
        )
        self.assertEqual(point["training_generation"], "autonomous-routes-v3")
        self.assertEqual(point["action_schema_version"], 3)
        self.assertEqual(point["outcome"]["hp_differential_bp"], 6000)
        cumulative = point["cumulative"]
        self.assertEqual(cumulative["rounds"], 3)
        self.assertEqual(cumulative["grammar_probe_attempts"], 4)
        self.assertEqual(cumulative["cancel_hypothesis_attempts"], 3)
        self.assertEqual(cumulative["cancel_edges"], 5)
        self.assertEqual(cumulative["connection_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

