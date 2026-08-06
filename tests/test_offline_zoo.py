from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_th105_policy_bundles import _coverage_summary, compare_distilled
from train_th105_zoo import _distilled_payload


def _record(episode: str, step: int, action: str) -> dict[str, object]:
    return {
        "episode_id": episode,
        "step": step,
        "difficulty": "lunatic",
        "opponent": "reimu@lunatic",
        "action": action,
        "state": {
            "relative_x_q4": 18,
            "relative_y_q4": 0,
            "self": {"hp_bp": 8000},
            "enemy": {"x_q4": 150, "hp_bp": 5000, "action": 300},
            "enemy_projectiles": [],
        },
    }


class OfflineZooTests(unittest.TestCase):
    def test_coverage_summary_counts_model_counterfactual_rows(self) -> None:
        summary = _coverage_summary(
            {
                "contexts": {
                    "ctx": {
                        "guard": {"support": 5, "factual_support": 5},
                        "jump": {"support": 5, "factual_support": 1},
                    }
                }
            }
        )
        self.assertEqual(summary["multi_action_contexts"], 1)
        self.assertEqual(summary["counterfactual_context_actions"], 1)

    def test_distillation_uses_support_and_mean_predictions(self) -> None:
        records = [_record("a", 0, "236B"), _record("a", 1, "236B")]
        result = _distilled_payload(
            records,
            {"damage_bp": [100.0, 300.0]},
            corpus_manifest={"statistics": {"difficulties": ["lunatic"]}},
            min_support=2,
        )
        contexts = result["contexts"]
        self.assertIsInstance(contexts, dict)
        assert isinstance(contexts, dict)
        entry = next(iter(contexts.values()))["236B"]
        self.assertEqual(entry["support"], 2)
        self.assertEqual(entry["outcomes"]["damage_bp"], 200.0)

    def test_policy_comparison_reports_top_action_disagreement(self) -> None:
        left = {
            "contexts": {
                "ctx": {
                    "A": {"outcomes": {"damage_bp": 500}},
                    "B": {"outcomes": {"damage_bp": 100}},
                }
            }
        }
        right = {
            "contexts": {
                "ctx": {
                    "A": {"outcomes": {"damage_bp": 100}},
                    "B": {"outcomes": {"damage_bp": 500}},
                }
            }
        }
        result = compare_distilled(left, right)
        self.assertEqual(result["common_multi_action_contexts"], 1)
        self.assertEqual(result["top_action_disagreements"], 1)
        self.assertEqual(result["top_action_agreement_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
