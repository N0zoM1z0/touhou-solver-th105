from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.battle import (
    BOOTSTRAP_CYCLE,
    FIGHTER_ACTION_ID,
    FIGHTER_HP,
    FIGHTER_POSITION_X,
    FRAME_DATA_BODY_BOX,
    FRAME_DATA_ATTACK_FLAGS,
    FRAME_DATA_BOX_VECTOR_A,
    FRAME_DATA_BOX_VECTOR_B,
    TerminalRoundTracker,
    boxes_world_envelope,
    cold_start_offense_prior,
    local_box_to_world,
    read_frame_box_vector,
    round_end_confirm_keys,
)


class FakeReader:
    def __init__(self, dwords: dict[int, int]) -> None:
        self.dwords = dwords

    def u32(self, address: int) -> int:
        return self.dwords[address] & 0xFFFFFFFF

    def i32(self, address: int) -> int:
        value = self.dwords[address] & 0xFFFFFFFF
        return value - 0x100000000 if value & 0x80000000 else value


class BootstrapBattleTests(unittest.TestCase):
    def test_cycle_exercises_movement_guard_and_attacks(self) -> None:
        chords = {frozenset(step.keys) for step in BOOTSTRAP_CYCLE}
        self.assertIn(frozenset({"right"}), chords)
        self.assertIn(frozenset({"left"}), chords)
        self.assertIn(frozenset({"down", "left"}), chords)
        self.assertIn(frozenset({"z"}), chords)
        self.assertIn(frozenset({"x"}), chords)

    def test_observed_fighter_offsets(self) -> None:
        self.assertEqual(FIGHTER_POSITION_X, 0xEC)
        self.assertEqual(FIGHTER_ACTION_ID, 0x13C)
        self.assertEqual(FIGHTER_HP, 0x174)
        self.assertEqual(FRAME_DATA_BODY_BOX, 0x54)
        self.assertEqual(FRAME_DATA_ATTACK_FLAGS, 0x50)
        self.assertEqual(FRAME_DATA_BOX_VECTOR_A, 0x5C)
        self.assertEqual(FRAME_DATA_BOX_VECTOR_B, 0x6C)

    def test_frame_box_vector_is_bounded_and_decoded(self) -> None:
        frame, begin = 0x10000, 0x20000
        reader = FakeReader(
            {
                frame + 0x5C: begin,
                frame + 0x60: begin + 32,
                frame + 0x64: begin + 32,
                begin + 0: -20,
                begin + 4: -180,
                begin + 8: 20,
                begin + 12: -80,
                begin + 16: -30,
                begin + 20: -80,
                begin + 24: 30,
                begin + 28: -1,
            }
        )
        self.assertEqual(
            read_frame_box_vector(reader, frame, 0x5C),
            ((-20, -180, 20, -80), (-30, -80, 30, -1)),
        )

    def test_local_box_transform_tracks_facing_and_vertical_origin(self) -> None:
        box = (-20, -180, 40, -20)
        self.assertEqual(
            local_box_to_world(box, x=100.0, y=10.0, facing=1),
            (80.0, 30.0, 140.0, 190.0),
        )
        self.assertEqual(
            local_box_to_world(box, x=100.0, y=10.0, facing=-1),
            (60.0, 30.0, 120.0, 190.0),
        )
        self.assertEqual(
            boxes_world_envelope((box,), x=100.0, y=10.0, facing=1),
            (80.0, 30.0, 140.0, 190.0),
        )

    def test_terminal_tracker_deduplicates_rebuilt_battle_loops(self) -> None:
        tracker = TerminalRoundTracker()
        self.assertIsNone(tracker.observe(me_hp=10000, enemy_hp=10000))
        self.assertEqual(tracker.observe(me_hp=0, enemy_hp=7400), 1)
        self.assertIsNone(tracker.observe(me_hp=0, enemy_hp=7400))
        self.assertIsNone(tracker.observe(me_hp=0, enemy_hp=0))
        self.assertIsNone(tracker.observe(me_hp=10000, enemy_hp=10000))
        self.assertEqual(tracker.observe(me_hp=2100, enemy_hp=0), 2)

    def test_terminal_tracker_ignores_cold_attach_to_dead_screen(self) -> None:
        tracker = TerminalRoundTracker()
        self.assertIsNone(tracker.observe(me_hp=0, enemy_hp=5000))

    def test_cold_start_prior_keeps_actions_but_not_round_attribution(self) -> None:
        original = {
            "*": {"236B": {"trials": 4, "connections": 2}},
            "__reward__": {
                "version": 1,
                "rounds": {"rounds": 3, "wins": 1, "losses": 2},
            },
        }
        seeded = cold_start_offense_prior(original)
        self.assertEqual(seeded["*"]["236B"]["trials"], 4)
        self.assertEqual(seeded["__reward__"]["rounds"]["rounds"], 0)
        self.assertEqual(original["__reward__"]["rounds"]["rounds"], 3)

    def test_round_end_confirm_uses_sparse_z_for_arena_dialogue(self) -> None:
        self.assertEqual(round_end_confirm_keys(0), {"z"})
        self.assertEqual(round_end_confirm_keys(1), set())
        self.assertEqual(round_end_confirm_keys(29), set())
        self.assertEqual(round_end_confirm_keys(30), {"z"})


if __name__ == "__main__":
    unittest.main()
