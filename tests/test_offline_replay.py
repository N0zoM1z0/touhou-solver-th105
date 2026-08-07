from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offline_replay import (
    native_opponent_key,
    replay_action_label,
    replay_transitions,
)


def _record(action: str = "motion-probe:236B") -> dict[str, object]:
    return {
        "transition_id": "episode:000001",
        "session_id": "session",
        "episode_id": "episode",
        "step": 1,
        "opponent": "0xENEMY@lunatic",
        "action": action,
        "combat_option": "attack",
        "duration_frames": 12,
        "state": {
            "relative_x_q4": 20,
            "self": {
                "character_vtable": "0x006B0EBC",
                "x_q4": 50,
                "y_q4": 0,
                "action": 0,
                "sequence": 0,
                "pose": 0,
            },
            "enemy": {
                "character_vtable": "0x006B165C",
                "x_q4": 70,
                "y_q4": 0,
                "action": 310,
                "action_frame": 4,
                "attack_box_count": 1,
            },
            "own_projectiles": [],
        },
        "next_state": {
            "self": {"x_q4": 60, "action": 310},
            "own_projectiles": [object()],
        },
        "outcome": {
            "damage": 600,
            "damage_bp": 600,
            "self_damage": 0,
            "self_damage_bp": 0,
            "spirit_delta": -200,
            "spirit_cost_bp": 2000,
            "punished": False,
            "terminal": None,
        },
    }


class OfflineReplayTests(unittest.TestCase):
    def test_native_row_identity_overrides_stale_shell_opponent(self) -> None:
        record = _record()
        record["difficulty"] = "lunatic"
        self.assertEqual(native_opponent_key(record), "0x006B165C@lunatic")

    def test_intent_maps_to_runtime_outcome_label(self) -> None:
        self.assertEqual(
            replay_action_label(_record()),
            "motion:0x006B0EBC:236B",
        )

    def test_replay_reuses_corpus_for_exact_action_priors(self) -> None:
        result = replay_transitions(
            [_record(), _record()],
            {"characters": {}},
            self_character="0x006B0EBC",
        )
        # Duplicate transition IDs are intentionally folded once.
        self.assertEqual(result.offense_samples, 1)
        self.assertEqual(result.option_samples, 1)
        self.assertEqual(result.rerouted_transitions, 1)
        state = result.opponents["0x006B165C@unknown"]["offense_outcomes"]
        exact_contexts = [key for key in state if "|ea=310" in key]
        self.assertEqual(len(exact_contexts), 1)
        self.assertIn("|sz=field", exact_contexts[0])
        row = state[exact_contexts[0]]
        self.assertEqual(row["motion:0x006B0EBC:236B"]["trials"], 1)


if __name__ == "__main__":
    unittest.main()
