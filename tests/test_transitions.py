from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.policy_api import PolicyDecision
from th105.transitions import OptionTransitionRecorder


class MemoryWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, record: dict[str, object]) -> None:
        self.rows.append(record)


class BulkMemoryWriter(MemoryWriter):
    def write_many(self, records: object) -> None:
        self.rows.extend(records)


def fighter(**changes: object) -> SimpleNamespace:
    values = {
        "vtable": 0x1234,
        "x": 100.0,
        "y": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "facing": 1,
        "action_id": 0,
        "action_sequence": 0,
        "action_pose": 0,
        "action_frame": 0,
        "frame_data_flags": 0,
        "attack_flags": 0,
        "hp": 10000,
        "max_hp": 10000,
        "spirit": 1000,
        "collision_state": 0,
        "hurt_boxes": (),
        "attack_boxes": (),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def battle(*, me: SimpleNamespace | None = None, enemy: SimpleNamespace | None = None):
    return SimpleNamespace(
        p1=me or fighter(),
        p2=enemy or fighter(x=500.0, vtable=0x5678),
    )


class OptionTransitionRecorderTests(unittest.TestCase):
    def recorder(self, writer: MemoryWriter) -> OptionTransitionRecorder:
        return OptionTransitionRecorder(
            writer,
            session_id="session",
            opponent="0x5678@hard",
            difficulty="hard",
            offline_policy_sha256="c" * 64,
        )

    def test_intent_boundary_records_raw_outcome_and_rle_keys(self) -> None:
        writer = MemoryWriter()
        recorder = self.recorder(writer)
        guard = PolicyDecision(frozenset({"left"}), "guard")
        attack = PolicyDecision(frozenset({"right", "z"}), "attack")
        recorder.observe(frame=0, state=battle(), decision=guard)
        recorder.observe(frame=1, state=battle(), decision=guard)
        changed = battle(
            me=fighter(hp=9950, spirit=900),
            enemy=fighter(x=500.0, vtable=0x5678, hp=9700),
        )

        recorder.observe(frame=2, state=changed, decision=attack)

        self.assertEqual(len(writer.rows), 1)
        row = writer.rows[0]
        self.assertEqual(row["action"], "guard")
        self.assertEqual(row["duration_frames"], 2)
        self.assertEqual(row["key_trace_rle"], [{"keys": "left", "frames": 2}])
        self.assertEqual(row["outcome"]["damage_bp"], 300)
        self.assertEqual(row["outcome"]["self_damage_bp"], 50)
        self.assertEqual(row["outcome"]["spirit_cost_bp"], 1000)
        self.assertIsNone(row["legal_actions"])
        self.assertFalse(row["legal_actions_known"])
        self.assertEqual(row["offline_policy_sha256"], "c" * 64)

    def test_terminal_closes_episode_and_next_observation_gets_new_id(self) -> None:
        writer = MemoryWriter()
        recorder = self.recorder(writer)
        decision = PolicyDecision(
            frozenset({"z"}),
            "attack",
            legal_actions=("attack", "guard"),
            behavior_probability=1.0,
        )
        initial = battle()
        terminal = battle(enemy=fighter(x=500.0, vtable=0x5678, hp=0))
        recorder.observe(frame=5, state=initial, decision=decision)
        recorder.finish_terminal(frame=20, state=terminal, won=True)
        first_episode = writer.rows[0]["episode_id"]
        recorder.observe(frame=30, state=initial, decision=decision)
        recorder.finish_truncated(frame=31, state=initial)

        self.assertEqual(writer.rows[0]["outcome"]["terminal"], "win")
        self.assertEqual(writer.rows[0]["legal_actions"], ["attack", "guard"])
        self.assertNotEqual(writer.rows[1]["episode_id"], first_episode)
        self.assertTrue(writer.rows[1]["outcome"]["truncated"])

    def test_policy_reload_is_an_explicit_boundary(self) -> None:
        writer = MemoryWriter()
        recorder = self.recorder(writer)
        decision = PolicyDecision(frozenset(), "watch")
        recorder.observe(
            frame=0, state=battle(), decision=decision, policy_generation=1
        )
        recorder.observe(
            frame=4, state=battle(), decision=decision, policy_generation=2
        )

        self.assertEqual(len(writer.rows), 1)
        self.assertEqual(writer.rows[0]["policy_generation"], 1)
        self.assertEqual(writer.rows[0]["duration_frames"], 4)

    def test_bulk_writer_flushes_at_episode_boundary(self) -> None:
        writer = BulkMemoryWriter()
        recorder = self.recorder(writer)
        recorder.observe(
            frame=0,
            state=battle(),
            decision=PolicyDecision(frozenset(), "watch"),
        )
        recorder.observe(
            frame=2,
            state=battle(),
            decision=PolicyDecision(frozenset({"left"}), "guard"),
        )
        self.assertEqual(writer.rows, [])

        recorder.finish_truncated(frame=4, state=battle())

        self.assertEqual(len(writer.rows), 2)


if __name__ == "__main__":
    unittest.main()
