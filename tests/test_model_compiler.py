from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.model_compiler import compile_knowledge


class ModelCompilerTests(unittest.TestCase):
    def test_compiles_response_and_temporal_lookup(self) -> None:
        compiled = compile_knowledge(
            {
                "version": 1,
                "characters": {
                    "0x01": {
                        "defense_responses": {
                            "302:0": {
                                "high_guard": {
                                    "trials": 1,
                                    "damage_events": 1,
                                    "total_damage": 500,
                                },
                                "low_guard": {"trials": 3, "successes": 3},
                            }
                        },
                        "projectile_envelopes": {"extents": {"810": 48.0}},
                        "profiles": {
                            "500": {
                                "completions": 4,
                                "mean_total": 32.0,
                                "projectile_samples": 2,
                                "first_projectile": 9.0,
                            }
                        },
                        "offense_outcomes": {
                            "close:ground:field:recovery": {
                                "AAAA": {
                                    "trials": 2,
                                    "connections": 2,
                                    "total_damage": 3000,
                                    "total_self_damage": 0,
                                    "total_spirit_cost": 0,
                                    "total_commitment": 80,
                                },
                                "slow": {
                                    "trials": 2,
                                    "connections": 0,
                                    "total_damage": 0,
                                    "total_self_damage": 500,
                                    "total_spirit_cost": 400,
                                    "total_commitment": 120,
                                },
                            }
                        },
                    }
                },
            }
        )
        character = compiled["characters"]["0x01"]
        self.assertEqual(
            character["defense_choices"]["302:0"]["response"], "low_guard"
        )
        self.assertEqual(character["projectile_extents"]["810"], 48.0)
        self.assertEqual(character["temporal_actions"]["500"]["danger_start"], 9.0)
        self.assertEqual(
            character["offense_choices"]["close:ground:field:recovery"]["action"],
            "AAAA",
        )


if __name__ == "__main__":
    unittest.main()
