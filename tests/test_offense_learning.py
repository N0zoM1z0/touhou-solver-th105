from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offense_learning import ActionOutcomeModel, combat_context


class ActionOutcomeTests(unittest.TestCase):
    def test_records_hit_startup_damage_and_punish(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "236B", "far:ground:field:recovery", frame=10, commitment=20,
            enemy_hp=10000, me_hp=10000, spirit=1000,
        )
        model.observe(frame=16, enemy_hp=9700, me_hp=10000, spirit=800)
        model.observe(frame=30, enemy_hp=9100, me_hp=9800, spirit=800)
        stats = model.table["*"]["236B"]
        self.assertEqual((stats.trials, stats.connections), (1, 1))
        self.assertEqual((stats.total_damage, stats.total_self_damage), (900, 200))
        self.assertEqual(stats.average_startup, 6.0)

    def test_untried_safe_punish_gets_finite_exploration_bonus(self) -> None:
        model = ActionOutcomeModel()
        self.assertEqual(model.adjustment("combo", "ctx", explore=True), 900.0)

    def test_failed_action_is_downranked(self) -> None:
        model = ActionOutcomeModel()
        for index in range(6):
            model.begin(
                "slow", "ctx", frame=index * 20, commitment=10,
                enemy_hp=10000, me_hp=10000, spirit=1000,
            )
            model.observe(
                frame=index * 20 + 10,
                enemy_hp=10000,
                me_hp=9500,
                spirit=800,
            )
        self.assertLess(model.adjustment("slow", "ctx"), -900.0)

    def test_context_is_bounded_categorical(self) -> None:
        self.assertEqual(
            combat_context(distance=50, enemy_y=0, enemy_x=50, phase="recovery"),
            "close:ground:corner:recovery",
        )

    def test_round_trip(self) -> None:
        first = ActionOutcomeModel()
        first.begin(
            "AAAA", "ctx", frame=0, commitment=5,
            enemy_hp=1000, me_hp=1000, spirit=1000,
        )
        first.observe(frame=5, enemy_hp=500, me_hp=1000, spirit=1000)
        second = ActionOutcomeModel()
        second.import_state(first.export_state())
        self.assertEqual(second.table["ctx"]["AAAA"].total_damage, 500)
        self.assertEqual(second.table["ctx"]["AAAA"].normalized_samples, 1)

    def test_terminal_credit_is_decayed_and_round_summary_is_persisted(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "safe", "ctx", frame=0, commitment=5,
            enemy_hp=10000, me_hp=10000, spirit=1000,
        )
        model.observe(frame=5, enemy_hp=9700, me_hp=10000, spirit=1000)
        model.observe_terminal(
            frame=125, won=True, enemy_hp=0, me_hp=4000, spirit=500,
        )
        stats = model.table["ctx"]["safe"]
        self.assertGreater(stats.terminal_win_credit, 0.0)
        restored = ActionOutcomeModel()
        restored.import_state(model.export_state())
        self.assertEqual(restored.rounds.wins, 1)

    def test_old_raw_checkpoint_remains_usable(self) -> None:
        model = ActionOutcomeModel()
        model.import_state({"ctx": {"legacy": {"trials": 2, "total_damage": 600}}})
        self.assertEqual(model.table["ctx"]["legacy"].normalized_samples, 0)
        self.assertGreater(model.adjustment("legacy", "ctx"), -1200.0)


if __name__ == "__main__":
    unittest.main()
