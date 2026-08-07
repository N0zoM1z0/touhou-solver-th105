from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offline_artifact import (
    DistilledOutcomePolicy,
    live_distillation_context,
    predicted_outcome_utility,
)


class OfflineArtifactTests(unittest.TestCase):
    def test_live_context_matches_training_buckets(self) -> None:
        me = SimpleNamespace(x=100.0, y=0.0, hp=7800, max_hp=10000)
        enemy = SimpleNamespace(
            x=172.0, y=0.0, hp=4200, max_hp=10000, action_id=308
        )
        state = SimpleNamespace(p1=me, p2=enemy)
        context = live_distillation_context(
            difficulty="lunatic",
            opponent="reimu@lunatic",
            state=state,
            enemy_projectiles=(object(),),
        )
        self.assertEqual(
            context,
            "lunatic:reimu@lunatic:close:ground:field:left-wall:ea308:p1-3:h7-4",
        )

    def test_outcome_scalarization_preserves_safety_bias(self) -> None:
        safe = predicted_outcome_utility(
            {
                "damage_bp": 600,
                "connection_probability": 1.0,
                "self_damage_bp": 0,
                "self_damage_p90_bp": 0,
                "commitment_frames": 30,
            }
        )
        trade = predicted_outcome_utility(
            {
                "damage_bp": 600,
                "connection_probability": 1.0,
                "self_damage_bp": 500,
                "self_damage_p90_bp": 1000,
                "commitment_frames": 30,
            }
        )
        self.assertGreater(safe, trade)

    def test_lookup_falls_back_on_unknown_and_scales_by_support(self) -> None:
        data = {
            "artifact_schema_version": 1,
            "contexts": {
                "ctx": {
                    "236B": {
                        "support": 10,
                        "outcomes": {
                            "damage_bp": 800,
                            "connection_probability": 1.0,
                        },
                    }
                }
            },
        }
        policy = DistilledOutcomePolicy(data)
        score = policy.score("ctx", "236B")
        self.assertIsNotNone(score)
        assert score is not None
        self.assertAlmostEqual(score.adjustment, score.utility * 0.5)
        self.assertIsNone(policy.score("ctx", "214B"))

    def test_bounded_backoff_stays_with_same_difficulty_opponent_and_action(self) -> None:
        source = "lunatic:reimu@lunatic:far:ground:field:ea200:p0:h9-5"
        data = {
            "artifact_schema_version": 1,
            "contexts": {
                source: {
                    "space-control-214C": {
                        "support": 4,
                        "outcomes": {"damage_bp": 500},
                    }
                },
                "lunatic:marisa@lunatic:far:ground:field:ea200:p0:h10-8": {
                    "space-control-214C": {
                        "support": 100,
                        "outcomes": {"damage_bp": 9000},
                    }
                },
            },
        }
        policy = DistilledOutcomePolicy(data)
        requested = "lunatic:reimu@lunatic:far:ground:field:ea200:p1-3:h10-8"
        score = policy.score(requested, "space-control-214C")
        self.assertIsNotNone(score)
        assert score is not None
        self.assertEqual(score.context, requested)
        self.assertEqual(score.matched_context, source)
        self.assertGreater(score.backoff_distance, 0.0)
        self.assertEqual(policy.metrics()["backoff_hits"], 1)
        self.assertIsNone(
            policy.score(requested.replace("reimu", "sakuya"), "space-control-214C")
        )

    def test_backoff_rejects_distant_state(self) -> None:
        data = {
            "artifact_schema_version": 1,
            "contexts": {
                "lunatic:reimu@lunatic:close:air:corner:ea500:p9+:h0-0": {
                    "space-control-214C": {
                        "support": 10,
                        "outcomes": {"damage_bp": 500},
                    }
                }
            },
        }
        policy = DistilledOutcomePolicy(data)
        self.assertIsNone(
            policy.score(
                "lunatic:reimu@lunatic:far:ground:field:ea200:p0:h10-10",
                "space-control-214C",
            )
        )

    def test_loader_rejects_incompatible_game_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_schema_version": 1,
                        "compatibility": {"game_build_sha256": ["good"]},
                        "contexts": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                DistilledOutcomePolicy.load(path, game_build_sha256="bad")


if __name__ == "__main__":
    unittest.main()
