from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.policies.adaptive import (
    AutonomousAdaptivePolicy,
    _bounded_bandit_score,
    _demonstration_frames,
    _demonstration_probe_frames,
    _hazard_projectiles,
    _human_demonstration_utility,
    _pattern_has_advancing_attack,
    _native_guard_response,
    _prune_legacy_coverage_scopes,
)
from th105.motion import generated_attack_probe_hypotheses, generated_motion_hypotheses
from th105.offense_learning import ActionOutcomeModel
from th105.battle import BattleState, FighterState
from th105.policy_api import PolicyObservation


def _fighter(*, x: float, vtable: int, facing: int) -> FighterState:
    return FighterState(
        pointer=1,
        vtable=vtable,
        x=x,
        y=0.0,
        velocity_x=0.0,
        velocity_y=0.0,
        facing=facing,
        action_id=0,
        action_sequence=0,
        action_pose=0,
        action_frame=0,
        frame_data_flags=0,
        attack_flags=0,
        hp=10000,
        max_hp=10000,
        collision_state=0,
        spirit=1000,
        command_mask=0,
        frame_data=0,
        body_box=None,
        hurt_boxes=(),
        attack_boxes=(),
    )


class AdaptiveNativeGeometryTests(unittest.TestCase):
    def test_coverage_migration_drops_full_context_scopes(self) -> None:
        policy = AutonomousAdaptivePolicy()
        explorer = policy.coverage_explorer
        explorer.selected["attack:lunatic:opponent:far|x"] = 3
        explorer.selected["attack:far:ground:neutral|x"] = 2
        explorer.opportunities.update(explorer.selected)
        explorer.scope_decisions["attack:lunatic:opponent:far"] = 3
        explorer.scope_decisions["attack:far:ground:neutral"] = 2
        _prune_legacy_coverage_scopes(explorer)
        self.assertEqual(explorer.selected, {"attack:far:ground:neutral|x": 2})
        self.assertEqual(explorer.scope_decisions, {"attack:far:ground:neutral": 2})

    def test_neutral_decision_exposes_complete_native_gated_set(self) -> None:
        policy = AutonomousAdaptivePolicy()
        state = BattleState(
            manager=1,
            p1=_fighter(x=200.0, vtable=0x006B0924, facing=1),
            p2=_fighter(x=600.0, vtable=0x006B18DC, facing=-1),
        )
        decision = policy.decide(
            PolicyObservation(
                frame=0,
                state=state,
                previous_state=None,
                enemy_projectiles=(),
                difficulty="lunatic",
                opponent_key="0x006B18DC@lunatic",
                exploration_rate=0.0,
            )
        )
        self.assertIn(decision.intent, decision.legal_actions or ())
        self.assertGreaterEqual(len(decision.legal_actions or ()), 2)
        self.assertEqual(decision.behavior_probability, 1.0)
        self.assertIsNotNone(decision.combat_option)
        self.assertIn(decision.combat_option, decision.legal_combat_options or ())
        self.assertGreaterEqual(len(decision.legal_combat_options or ()), 3)
        self.assertIn("|ef=neutral", decision.learning_context or "")

    def test_offensive_enemy_action_conditions_the_learning_context(self) -> None:
        policy = AutonomousAdaptivePolicy()
        enemy = replace(
            _fighter(x=260.0, vtable=0x006B18DC, facing=-1),
            action_id=310,
            attack_boxes=((0, 0, 20, 20),),
        )
        observation = PolicyObservation(
            frame=10,
            state=BattleState(
                manager=1,
                p1=_fighter(x=200.0, vtable=0x006B0EBC, facing=1),
                p2=enemy,
            ),
            previous_state=None,
            enemy_projectiles=(),
            difficulty="lunatic",
            opponent_key="0x006B18DC@lunatic",
            exploration_rate=0.0,
        )
        decision = policy.decide(observation)
        self.assertIn("|ef=strike", decision.learning_context or "")
        self.assertIn("|ea=310", decision.learning_context or "")

    def test_frame_attack_box_replaces_learned_square_extent(self) -> None:
        projectile = SimpleNamespace(
            x=100.0,
            y=50.0,
            velocity_x=-4.0,
            velocity_y=2.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            facing=-1,
            attack_boxes=((-30, -10, 10, 20),),
        )
        hazard = _hazard_projectiles(projectile, half_extent=99.0)[0]
        self.assertEqual((hazard.x, hazard.y), (110.0, 45.0))
        self.assertEqual((hazard.half_width, hazard.half_height), (20.0, 15.0))

    def test_empty_native_attack_boxes_mean_inactive_frame(self) -> None:
        projectile = SimpleNamespace(
            x=0.0,
            y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            attack_boxes=(),
        )
        self.assertEqual(_hazard_projectiles(projectile, half_extent=99.0), ())

    def test_unambiguous_native_guard_level_overrides_learning(self) -> None:
        self.assertEqual(_native_guard_response(0x2, "low_guard"), "high_guard")
        self.assertEqual(_native_guard_response(0x4, "high_guard"), "low_guard")
        self.assertEqual(_native_guard_response(0x6, "high_guard"), "high_guard")

    def test_real_damage_can_override_native_guard_prior(self) -> None:
        self.assertEqual(
            _native_guard_response(0x2, "low_guard", native_failed=True),
            "low_guard",
        )
        self.assertEqual(
            _native_guard_response(0x4, "backdash", native_failed=True),
            "backdash",
        )

    def test_human_pattern_replays_relative_to_current_facing(self) -> None:
        frames = _demonstration_frames(
            "down@1,down+toward@2,toward+x@1,neutral@2", "left"
        )
        self.assertEqual(frames[0], {"down"})
        self.assertEqual(frames[1], {"down", "left"})
        self.assertEqual(frames[3], {"left", "x"})
        self.assertEqual(frames[-1], set())

    def test_human_pattern_rejects_unbounded_or_unknown_input(self) -> None:
        self.assertEqual(_demonstration_frames("toward+q@2", "right"), [])
        self.assertEqual(_demonstration_frames("z@181", "right"), [])

    def test_advancing_attack_accepts_chord_or_short_sequence(self) -> None:
        self.assertTrue(_pattern_has_advancing_attack("a+toward+x@3"))
        self.assertTrue(_pattern_has_advancing_attack("toward@3,up+z@2"))
        self.assertFalse(_pattern_has_advancing_attack("back@3,x@2"))
        self.assertFalse(_pattern_has_advancing_attack("toward@8,z@2"))

    def test_neutral_probe_extracts_only_approach_attack_edge(self) -> None:
        frames = _demonstration_probe_frames(
            "back@5,toward@3,a+toward+x@3,neutral@8", "right"
        )
        self.assertTrue(frames)
        self.assertLessEqual(len(frames), 12)
        self.assertNotIn("left", set().union(*frames))
        self.assertIn({"a", "right", "x"}, frames)

    def test_autonomous_probes_are_short_bounded_attacks(self) -> None:
        for _label, pattern in generated_attack_probe_hypotheses():
            frames = _demonstration_probe_frames(pattern, "right")
            self.assertTrue(frames)
            self.assertLessEqual(len(frames), 12)
            self.assertTrue(any(keys & {"z", "x", "c"} for keys in frames))

    def test_cancel_discovery_exposes_complete_generic_followup_set(self) -> None:
        policy = AutonomousAdaptivePolicy()
        me = replace(
            _fighter(x=200.0, vtable=0x006B0924, facing=1),
            action_id=300,
            action_sequence=0,
            action_pose=2,
            action_frame=8,
        )
        state = BattleState(
            manager=1,
            p1=me,
            p2=_fighter(x=260.0, vtable=0x006B18DC, facing=-1),
        )
        selected = policy._start_cancel_hypothesis(
            PolicyObservation(
                frame=10,
                state=state,
                previous_state=None,
                enemy_projectiles=(),
                difficulty="lunatic",
                opponent_key="0x006B18DC@lunatic",
                exploration_rate=0.0,
            ),
            "right",
        )
        self.assertIsNotNone(selected)
        legal = policy.queue_legal_actions or ()
        self.assertEqual(len(legal), 69)
        self.assertTrue(any(name.endswith("chord:down+toward+x") for name in legal))
        self.assertTrue(any(name.endswith("motion:236B") for name in legal))

    def test_patchouli_uses_the_complete_generic_motion_catalog(self) -> None:
        policy = AutonomousAdaptivePolicy()
        state = BattleState(
            manager=1,
            p1=_fighter(x=200.0, vtable=0x006B0EBC, facing=1),
            p2=_fighter(x=600.0, vtable=0x006B18DC, facing=-1),
        )
        selected = policy._start_motion_hypothesis(
            PolicyObservation(
                frame=10,
                state=state,
                previous_state=None,
                enemy_projectiles=(),
                difficulty="lunatic",
                opponent_key="0x006B18DC@lunatic",
                exploration_rate=0.0,
            ),
            "right",
        )
        self.assertIsNotNone(selected)
        legal = policy.queue_legal_actions or ()
        self.assertEqual(len(legal), len(generated_motion_hypotheses()))
        self.assertIn("motion-probe:236B", legal)
        self.assertIn("motion-probe:214C", legal)

    def test_close_pressure_uses_the_generic_chord_catalog(self) -> None:
        policy = AutonomousAdaptivePolicy()
        state = BattleState(
            manager=1,
            p1=_fighter(x=200.0, vtable=0x006B0EBC, facing=1),
            p2=_fighter(x=240.0, vtable=0x006B18DC, facing=-1),
        )
        decision = policy.decide(
            PolicyObservation(
                frame=10,
                state=state,
                previous_state=None,
                enemy_projectiles=(),
                difficulty="lunatic",
                opponent_key="0x006B18DC@lunatic",
                exploration_rate=0.0,
            )
        )
        legal = decision.legal_actions or ()
        self.assertEqual(len(legal), len(generated_attack_probe_hypotheses()))
        self.assertTrue(all(action.startswith("grammar-probe:") for action in legal))
        self.assertNotIn("close-pressure-z", legal)

    def test_bandit_prefers_proven_success_over_untried_novelty(self) -> None:
        model = ActionOutcomeModel()
        model.begin(
            "known",
            "ctx",
            frame=0,
            commitment=1,
            enemy_hp=10000,
            me_hp=10000,
            spirit=1000,
        )
        model.observe(frame=1, enemy_hp=9000, me_hp=10000, spirit=1000)
        self.assertGreater(
            _bounded_bandit_score(model, "known", "ctx"),
            _bounded_bandit_score(model, "untried", "ctx"),
        )

    def test_learned_melee_box_keeps_future_active_delay(self) -> None:
        policy = AutonomousAdaptivePolicy()
        geometry = policy.enemy_attack_geometry
        geometry.observe(0, action_id=305, sequence=0, pose=0, attack_boxes=())
        geometry.observe(
            5,
            action_id=305,
            sequence=0,
            pose=1,
            attack_boxes=((10, -80, 100, 0),),
        )
        geometry.observe(6, action_id=0, sequence=0, pose=0, attack_boxes=())
        geometry.observe(100, action_id=305, sequence=0, pose=0, attack_boxes=())
        enemy = SimpleNamespace(
            action_id=305,
            action_sequence=0,
            attack_boxes=(),
            x=500.0,
            y=0.0,
            facing=1,
            velocity_x=0.0,
            velocity_y=0.0,
        )
        observation = SimpleNamespace(frame=100, state=SimpleNamespace(p2=enemy))
        hazards = policy._enemy_melee_hazards(observation)
        self.assertEqual(len(hazards), 1)
        self.assertEqual(hazards[0].active_start_frame, 5)
        self.assertEqual(hazards[0].active_end_frame, 5)
        self.assertEqual((hazards[0].half_width, hazards[0].half_height), (45.0, 40.0))

    def test_hot_reload_state_is_newer_than_disk_priors(self) -> None:
        first = AutonomousAdaptivePolicy()
        first.enemy_attack_geometry.observe(
            0,
            action_id=305,
            sequence=0,
            pose=2,
            attack_boxes=((0, -80, 90, 0),),
        )
        first.opponent.observe(
            0,
            enemy_action=305,
            me_hp=10000,
            projectile_count=0,
            active_hitbox=True,
        )
        second = AutonomousAdaptivePolicy()
        second.import_state(first.export_state())
        self.assertIsNotNone(second.enemy_attack_geometry.forecast(305, 0))
        self.assertIn(305, second.opponent.profiles)
        self.assertTrue(second.opponent_knowledge_seeded)
        self.assertTrue(second.attack_geometry_knowledge_seeded)

    def test_losing_human_trade_is_not_a_replay_candidate(self) -> None:
        losing = {
            "trials": 1,
            "connections": 1,
            "total_damage": 480,
            "total_self_damage": 1125,
            "total_spirit_cost": 588,
            "total_duration": 70,
        }
        self.assertIsNone(
            _human_demonstration_utility(
                losing, spirit=1000, support=1, context_penalty=0.0
            )
        )

    def test_safe_human_hit_has_positive_prior_utility(self) -> None:
        safe = {
            "trials": 1,
            "connections": 1,
            "total_damage": 690,
            "total_self_damage": 0,
            "total_spirit_cost": 200,
            "total_duration": 52,
        }
        utility = _human_demonstration_utility(
            safe, spirit=1000, support=1, context_penalty=0.0
        )
        self.assertIsNotNone(utility)
        assert utility is not None
        self.assertGreater(utility, 0.0)


if __name__ == "__main__":
    unittest.main()
