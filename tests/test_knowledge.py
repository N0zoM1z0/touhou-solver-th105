from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.knowledge import (
    attack_geometry_for,
    cancel_graph_for,
    character_models_from_data,
    defense_model_for,
    offense_model_for,
    persist_character_models,
    persist_profiles,
    profiles_for,
    projectile_model_for,
)
from th105.opponent_model import OpponentActionModel


class OpponentKnowledgeTests(unittest.TestCase):
    def test_extracts_all_model_families_from_one_snapshot(self) -> None:
        models = character_models_from_data(
            {
                "characters": {
                    "opponent": {
                        "profiles": {"300": {"starts": 1}},
                        "offense_outcomes": {"*": {"z": {"trials": 2}}},
                        "cancel_graph": "invalid",
                    }
                }
            },
            "opponent",
        )

        self.assertEqual(models["profiles"]["300"]["starts"], 1)
        self.assertEqual(models["offense_outcomes"]["*"]["z"]["trials"], 2)
        self.assertEqual(models["cancel_graph"], {})
        self.assertEqual(models["defense_responses"], {})

    def test_snapshot_extraction_tolerates_invalid_roots(self) -> None:
        self.assertTrue(
            all(not value for value in character_models_from_data([], "x").values())
        )

    def test_round_trip_and_model_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            profiles = {
                "400": {
                    "starts": 3,
                    "completions": 2,
                    "mean_total": 30.0,
                    "impact_samples": 0,
                    "first_impact": 0.0,
                    "last_impact": 0.0,
                    "projectile_samples": 2,
                    "first_projectile": 8.0,
                    "last_projectile": 9.0,
                }
            }
            persist_profiles(path, "0x12345678", profiles)
            root = json.loads(path.read_text())
            self.assertEqual(root["schema_version"], 2)
            self.assertEqual(root["action_schema_version"], 4)
            self.assertEqual(root["training_generation"], "autonomous-grammar-v4")
            loaded = profiles_for(path, "0x12345678")
            model = OpponentActionModel()
            model.seed(loaded)
            self.assertEqual(model.profiles[400].completions, 2)
            self.assertEqual(model.profiles[400].first_projectile_frame, 8.0)

    def test_character_entry_keeps_both_model_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            persist_character_models(
                path,
                "0xABCDEF00",
                profiles={"500": {"starts": 2}},
                projectile_envelopes={
                    "extents": {"810": 54.0},
                    "samples": {"810": 3},
                },
                defense_responses={
                    "302:0": {"low_guard": {"trials": 2, "successes": 2}}
                },
                offense_outcomes={
                    "close:ground:field:recovery": {
                        "AAAA": {"trials": 2, "connections": 2}
                    }
                },
                attack_geometry={
                    "305:0": {
                        "action_id": 305,
                        "sequence": 0,
                        "active_observations": 4,
                    }
                },
                cancel_graph={
                    "300:0:1|z|305:0": {
                        "source_action": 300,
                        "target_action": 305,
                        "trials": 3,
                    }
                },
            )
            self.assertEqual(profiles_for(path, "0xABCDEF00")["500"]["starts"], 2)
            self.assertEqual(
                projectile_model_for(path, "0xABCDEF00")["extents"]["810"],
                54.0,
            )
            self.assertEqual(
                defense_model_for(path, "0xABCDEF00")["302:0"]["low_guard"][
                    "successes"
                ],
                2,
            )
            self.assertEqual(
                offense_model_for(path, "0xABCDEF00")[
                    "close:ground:field:recovery"
                ]["AAAA"]["connections"],
                2,
            )
            self.assertEqual(
                attack_geometry_for(path, "0xABCDEF00")["305:0"][
                    "active_observations"
                ],
                4,
            )
            self.assertEqual(
                cancel_graph_for(path, "0xABCDEF00")["300:0:1|z|305:0"][
                    "trials"
                ],
                3,
            )

    def test_profile_only_update_preserves_projectile_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            persist_character_models(
                path,
                "opponent",
                profiles={},
                projectile_envelopes={"extents": {"801": 40.0}},
            )
            persist_profiles(path, "opponent", {"400": {"starts": 1}})
            self.assertEqual(
                projectile_model_for(path, "opponent")["extents"]["801"], 40.0
            )


if __name__ == "__main__":
    unittest.main()
