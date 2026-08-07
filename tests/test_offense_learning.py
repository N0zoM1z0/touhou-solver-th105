from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offense_learning import ActionOutcomeModel, OptionOutcomeModel, combat_context


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

    def test_non_damage_motion_effect_is_preserved_as_compact_statistics(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "motion:character:22B",
            "ctx",
            frame=0,
            commitment=8,
            enemy_hp=10000,
            me_hp=10000,
            spirit=1000,
            me_x=100.0,
            me_action_id=0,
            projectile_count=0,
        )
        model.observe(
            frame=8,
            enemy_hp=10000,
            me_hp=10000,
            spirit=900,
            me_x=360.0,
            me_action_id=600,
            projectile_count=0,
        )
        stats = model.table["ctx"]["motion:character:22B"]
        self.assertEqual(stats.effectful_trials, 1)
        self.assertEqual(stats.action_change_trials, 1)
        self.assertEqual(stats.total_displacement, 260)

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
        self.assertIn("@family:AAAA", second.hierarchy["ctx"])

    def test_motion_hierarchy_shares_sparse_evidence(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "motion:character:236B", "ctx", frame=0, commitment=5,
            enemy_hp=10000, me_hp=10000, spirit=1000,
        )
        model.observe(frame=5, enemy_hp=9400, me_hp=10000, spirit=900)
        estimate = model.estimate("motion:other:236B", "ctx")
        self.assertGreater(estimate.effective_trials, 0)
        self.assertGreater(estimate.utility, 0)

    def test_projectile_receives_delayed_eligibility_credit(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "motion:character:236B", "ctx", frame=0, commitment=5,
            enemy_hp=10000, me_hp=10000, spirit=1000, projectile_count=0,
        )
        model.observe(
            frame=5, enemy_hp=10000, me_hp=10000, spirit=900,
            projectile_count=1,
        )
        model.observe(frame=45, enemy_hp=9500, me_hp=10000, spirit=900)
        stats = model.table["ctx"]["motion:character:236B"]
        self.assertGreater(stats.delayed_damage_credit_bp, 0.0)
        self.assertGreater(stats.eligibility_events, 0)

    def test_option_value_does_not_apply_attack_whiff_penalty(self) -> None:
        model = OptionOutcomeModel()
        model.begin(
            "option:defend", "ctx", frame=0, commitment=30,
            enemy_hp=10000, me_hp=10000, spirit=1000,
        )
        model.observe(frame=30, enemy_hp=10000, me_hp=10000, spirit=1000)
        self.assertEqual(model.estimate("option:defend", "ctx").utility, 0.0)

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
        self.assertEqual(model.table["ctx"]["legacy"].normalized_samples, 2)
        self.assertEqual(model.table["ctx"]["legacy"].total_damage_bp, 600)
        self.assertGreater(model.adjustment("legacy", "ctx"), -1200.0)

    def test_legacy_evidence_survives_a_new_normalized_whiff(self) -> None:
        model = ActionOutcomeModel()
        model.import_state(
            {
                "*": {
                    "legacy": {
                        "trials": 4,
                        "connections": 2,
                        "total_damage": 600,
                        "total_self_damage": 300,
                        "punished_trials": 1,
                        "total_spirit_cost": 20,
                    }
                }
            }
        )
        stats = model.table["*"]["legacy"]
        self.assertEqual(stats.normalized_samples, 4)
        self.assertEqual(stats.total_damage_bp, 600)
        self.assertEqual(stats.total_self_damage_bp, 300)
        self.assertEqual(stats.total_spirit_cost_bp, 200)
        self.assertEqual(sum(stats.self_damage_histogram), 4)
        model.begin(
            "legacy", "*", frame=10, commitment=1,
            enemy_hp=10000, me_hp=10000, spirit=1000,
        )
        model.observe(frame=11, enemy_hp=10000, me_hp=10000, spirit=1000)
        self.assertEqual(stats.normalized_samples, 5)
        self.assertEqual(stats.total_damage_bp, 600)


if __name__ == "__main__":
    unittest.main()
