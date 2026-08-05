"""Sakuya Easy-Arcade policy; edits to this file hot reload during combat."""

from __future__ import annotations

import math
import hashlib
import time
from collections import Counter, deque

from ..battle import boxes_world_envelope, local_box_to_world
from ..defense_learning import DefenseResponseModel
from ..hazard import HazardEvaluator, HazardProjectile, MovementCandidate
from ..human_learning import human_demonstration_utility as _human_demonstration_utility
from ..motion import build_motion_frames
from ..offense_learning import ActionOutcomeModel, combat_context
from ..opponent_model import ActionAssessment, OpponentActionModel
from ..policy_api import POLICY_API_VERSION, PolicyDecision, PolicyObservation
from ..projectile_model import ProjectileEnvelopeModel
from ..sakuya_combos import COMBOS, ComboContext, select_combo
from ..tactics import ActionCandidate, TacticalContext, rank_actions


SKILL_ACTIONS = {
    "236B": 500,
    "236C": 501,
    "214B": 520,
    "214C": 521,
    "623B": 540,
    "623C": 541,
}

# Keep the movement macros inside the policy plugin.  The resident battle shell
# deliberately has a tiny stable ABI; importing newly-added helpers from an
# already-cached module would otherwise make a policy-only hot reload fail.
def _dash_frames(direction: str, hold_frames: int = 10) -> list[set[str]]:
    return [{direction}] * 2 + [set()] * 2 + [{direction}] * hold_frames + [set()] * 3


def _jump_frames(
    toward: str,
    direction: str = "neutral",
    *,
    super_jump: bool = False,
    hold_frames: int = 10,
) -> list[set[str]]:
    horizontal = None
    if direction == "toward":
        horizontal = toward
    elif direction == "back":
        horizontal = "left" if toward == "right" else "right"
    jump = {"up"} | ({horizontal} if horizontal else set())
    prefix = [{"down"}] * 2 + [set()] if super_jump else []
    return prefix + [jump] * hold_frames + [set()] * 3


def _flight_frames(direction: set[str], hold_frames: int = 12) -> list[set[str]]:
    # Physical keyboard A maps to the game's D movement button, not a spell.
    return [set(direction) | {"a"}] * hold_frames + [set()] * 3


def _movement_candidate(
    velocity_x: float,
    velocity_y: float,
    *,
    half_width: float = 0.0,
    half_height: float = 0.0,
    graze_frames: int = 0,
    startup_frames: int = 0,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> MovementCandidate:
    """Remain hot-loadable in a shell that cached hazard ABI 1.

    A fresh shell gets the ABI-2 hitbox/graze model.  An old resident shell can
    still load this policy and use linear x/y candidates until its next safe
    restart.
    """
    try:
        return MovementCandidate(
            velocity_x,
            velocity_y,
            half_width=half_width,
            half_height=half_height,
            graze_frames=graze_frames,
            startup_frames=startup_frames,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
        )
    except TypeError:
        try:
            return MovementCandidate(
                velocity_x,
                velocity_y,
                half_height=half_height,
                graze_frames=graze_frames,
            )
        except TypeError:
            return MovementCandidate(velocity_x, velocity_y)


def _hazard_projectiles(
    projectile, half_extent: float = 32.0
) -> tuple[HazardProjectile, ...]:
    def sane(value: object, *, limit: float) -> float:
        value = float(value)
        return value if math.isfinite(value) and abs(value) <= limit else 0.0

    velocity_x = sane(projectile.velocity_x, limit=1000.0)
    velocity_y = sane(projectile.velocity_y, limit=1000.0)
    acceleration_x = sane(getattr(projectile, "acceleration_x", 0.0), limit=100.0)
    acceleration_y = sane(getattr(projectile, "acceleration_y", 0.0), limit=100.0)
    attack_boxes = getattr(projectile, "attack_boxes", None)
    if attack_boxes is not None:
        hazards: list[HazardProjectile] = []
        for box in attack_boxes:
            left, bottom, right, top = local_box_to_world(
                box,
                x=float(projectile.x),
                y=float(projectile.y),
                facing=int(getattr(projectile, "facing", 1)),
            )
            hazards.append(
                HazardProjectile(
                    (left + right) * 0.5,
                    (bottom + top) * 0.5,
                    velocity_x,
                    velocity_y,
                    half_width=(right - left) * 0.5,
                    half_height=(top - bottom) * 0.5,
                    acceleration_x=acceleration_x,
                    acceleration_y=acceleration_y,
                )
            )
        # Empty attack-box vectors are inactive animation frames, not hazards.
        return tuple(hazards)
    try:
        return (
            HazardProjectile(
                projectile.x,
                projectile.y,
                velocity_x,
                velocity_y,
                half_width=half_extent,
                half_height=half_extent,
                acceleration_x=acceleration_x,
                acceleration_y=acceleration_y,
            ),
        )
    except TypeError:
        return (HazardProjectile(projectile.x, projectile.y, velocity_x, velocity_y),)


def _reset_model_episode(model: OpponentActionModel) -> None:
    reset = getattr(model, "reset_episode", None)
    if callable(reset):
        reset()
        return
    # Compatibility for a resident shell that cached the previous helper
    # module before reset_episode was added.
    model.episode = None
    model.last_enemy_action = None
    model.last_me_hp = None
    model.last_projectile_count = 0


# Conservative total exposure windows, pending exact per-pose values from the
# Sakuya action tables.  They are used only to reject unsafe casts; once an
# action starts, native frame flags decide whether the agent is still committed.
SKILL_COMMIT_FRAMES = {
    "236B": 84,
    "236C": 90,
    "214B": 54,
    "214C": 60,
    "623B": 38,
    "623C": 42,
}

SKILL_FALLBACK_STARTUP = {"236B": 14.0, "236C": 16.0, "214B": 18.0, "214C": 20.0}
SKILL_DAMAGE = {"236B": 1200.0, "236C": 1100.0, "214B": 850.0, "214C": 900.0}
SKILL_INPUT = {
    "236B": ("236", "x", "strike"),
    "236C": ("236", "c", "spread"),
    "214B": ("214", "x", "setup"),
    "214C": ("214", "c", "setup"),
}


def _native_guard_response(attack_flags: int, learned: str) -> str:
    """Use frame-data guard type when it is unambiguous, else learned prior."""
    mid_hit = bool(attack_flags & 0x2)
    low_hit = bool(attack_flags & 0x4)
    if low_hit and not mid_hit:
        return "low_guard"
    if mid_hit and not low_hit:
        return "high_guard"
    return learned


def _demonstration_frames(pattern: str, toward: str) -> list[set[str]]:
    """Decode a bounded facing-normalized human input pattern."""
    back = "left" if toward == "right" else "right"
    mapping = {"toward": toward, "back": back}
    allowed = {"up", "down", "z", "x", "c", "a", "toward", "back"}
    frames: list[set[str]] = []
    for raw_run in pattern.split(",")[:64]:
        chord, separator, raw_count = raw_run.rpartition("@")
        if not separator:
            return []
        try:
            count = int(raw_count)
        except ValueError:
            return []
        if not 1 <= count <= 60:
            return []
        names = set() if chord == "neutral" else set(chord.split("+"))
        if not names <= allowed:
            return []
        keys = {mapping.get(name, name) for name in names}
        frames.extend([set(keys) for _ in range(count)])
        if len(frames) > 180:
            return []
    return frames


class SakuyaAdaptivePolicy:
    api_version = POLICY_API_VERSION
    name = "sakuya-adaptive-v1"

    def __init__(self) -> None:
        self.queue: deque[set[str]] = deque()
        self.evade_queue: deque[set[str]] = deque()
        self.attack_cooldown = 0
        self.skill_cooldown = 45
        self.evade_cooldown = 0
        self.skill_index = 0
        self.pending_skill = 0
        self.pending_skill_label: str | None = None
        self.active_skill_label: str | None = None
        self.active_skill_start_frame = 0
        self.active_skill_start_hp = 0
        self.last_enemy_hp: int | None = None
        self.last_me_hp: int | None = None
        self.last_me_action: int | None = None
        self.last_enemy_action: int | None = None
        self.combo_target_hp: int | None = None
        self.combo_confirm_deadline = 0
        self.counts: Counter[str] = Counter()
        self.actions: Counter[int] = Counter()
        self.enemy_actions: Counter[int] = Counter()
        self.hazard = HazardEvaluator()
        self.opponent = OpponentActionModel()
        self.own_action_model = OpponentActionModel()
        self.projectile_envelopes = ProjectileEnvelopeModel()
        self.defense_responses = DefenseResponseModel()
        self.offense_outcomes = ActionOutcomeModel()
        self.last_assessment = ActionAssessment(0, 0, "neutral", 0.0, 0.0, 0.0, False)
        self.last_tactical_scores: list[dict[str, float | str]] = []
        self.last_risk: dict[str, float | bool | int] = {}
        self.last_own_projectile_count = 0
        self.previous_intent = "initializing"
        self.opponent_knowledge_seeded = False
        self.projectile_knowledge_seeded = False
        self.defense_knowledge_seeded = False
        self.offense_knowledge_seeded = False
        self.human_knowledge_seeded = False
        self.human_demonstrations: dict[str, object] = {}
        self.last_human_demonstration: dict[str, object] = {}
        self.last_damage_frame = -10000
        self.last_damage_context: dict[str, object] = {}
        self.defense_motion_issued = False
        self.guard_chain_until = 0
        self.hazard_calls = 0
        self.hazard_total_ns = 0
        self.hazard_max_ns = 0

    def export_state(self) -> dict[str, object]:
        # Preserve learned counters/cooldowns, but never transplant a half-issued
        # input sequence across code generations.
        return {
            "attack_cooldown": self.attack_cooldown,
            "skill_cooldown": self.skill_cooldown,
            "skill_index": self.skill_index,
            "counts": dict(self.counts),
            "actions": dict(self.actions),
            "enemy_actions": dict(self.enemy_actions),
            "projectile_envelopes": self.projectile_envelopes.export_state(),
            "defense_responses": self.defense_responses.export_state(),
            "offense_outcomes": self.offense_outcomes.export_state(),
        }

    def import_state(self, state: dict[str, object]) -> None:
        self.attack_cooldown = int(state.get("attack_cooldown", 0))
        self.skill_cooldown = int(state.get("skill_cooldown", 0))
        self.skill_index = int(state.get("skill_index", 0))
        self.counts.update(state.get("counts", {}))
        self.actions.update(state.get("actions", {}))
        self.enemy_actions.update(state.get("enemy_actions", {}))
        self.projectile_envelopes.import_state(state.get("projectile_envelopes", {}))
        self.defense_responses.import_state(state.get("defense_responses", {}))
        self.offense_outcomes.import_state(state.get("offense_outcomes", {}))

    def _hazards(self, projectiles) -> tuple[HazardProjectile, ...]:
        return tuple(
            hazard
            for projectile in projectiles
            for hazard in _hazard_projectiles(
                projectile,
                self.projectile_envelopes.extent_for(projectile.action_id),
            )
        )

    def _decision(self, keys: set[str], intent: str) -> PolicyDecision:
        self.counts[f"intent:{intent}"] += 1
        self.previous_intent = intent
        return PolicyDecision(frozenset(keys), intent)

    def _evaluate_hazard(self, *args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return self.hazard.evaluate(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - started
            self.hazard_calls += 1
            self.hazard_total_ns += elapsed
            self.hazard_max_ns = max(self.hazard_max_ns, elapsed)

    def _start_combo(
        self,
        observation: PolicyObservation,
        toward: str,
        *,
        punish: bool,
    ) -> tuple[set[str], str] | None:
        me, enemy = observation.state.p1, observation.state.p2
        combo_context = ComboContext(
            distance=abs(enemy.x - me.x),
            enemy_x=enemy.x,
            enemy_y=enemy.y,
            spirit=int(getattr(me, "spirit", 1000)),
            punish=punish,
        )
        outcome_context = combat_context(
            distance=combo_context.distance,
            enemy_y=enemy.y,
            enemy_x=enemy.x,
            phase=self.last_assessment.phase,
        )
        adjustments = {
            candidate.name: self.offense_outcomes.adjustment(
                candidate.name, outcome_context, explore=punish
            )
            for candidate in COMBOS
        }
        try:
            combo = select_combo(
                combo_context,
                learned_adjustments=adjustments,
            )
        except TypeError:
            # A resident controller may have cached the pre-learning catalogue.
            combo = select_combo(combo_context)
        if combo is None:
            return None
        self.queue.extend(combo.builder(toward))
        self.offense_outcomes.begin(
            combo.name,
            outcome_context,
            frame=observation.frame,
            commitment=len(self.queue) + 18,
            enemy_hp=enemy.hp,
            me_hp=me.hp,
            spirit=int(getattr(me, "spirit", 1000)),
        )
        self.combo_target_hp = enemy.hp
        self.combo_confirm_deadline = observation.frame + 18
        self.attack_cooldown = 75
        prefix = "punish" if punish else "combo"
        self.counts[f"{prefix}_attempts:{combo.name}"] += 1
        return self.queue.popleft(), f"{prefix}:{combo.name}"

    def _start_human_demonstration(
        self,
        observation: PolicyObservation,
        toward: str,
    ) -> tuple[set[str], str] | None:
        me, enemy = observation.state.p1, observation.state.p2
        context = combat_context(
            distance=abs(enemy.x - me.x),
            enemy_y=enemy.y,
            enemy_x=enemy.x,
            phase="reaction",
        )
        viable: list[tuple[float, str, str, str, int]] = []
        spirit = int(getattr(me, "spirit", 1000))
        prefix = context.rsplit(":", 1)[0]
        context_options = ((context, 0.0), (f"{prefix}:neutral", 100.0))
        compiled_contexts = self.human_demonstrations.get("contexts", {})
        if isinstance(compiled_contexts, dict):
            for source_context, context_penalty in context_options:
                candidates = compiled_contexts.get(source_context, ())
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    average_spirit = float(candidate.get("average_spirit", 0.0))
                    if average_spirit > max(0, spirit - 200):
                        continue
                    score = float(candidate.get("score", 0.0)) - context_penalty
                    if score <= 0.0:
                        continue
                    viable.append(
                        (
                            score,
                            str(candidate.get("signature", "")),
                            str(candidate.get("pattern", "")),
                            str(candidate.get("pattern_id", "legacy")),
                            int(candidate.get("connections", 0)),
                        )
                    )
        else:
            # Compatibility for a controller started before the offline human
            # compiler existed. Fresh encounters receive only compiled rows.
            chains = self.human_demonstrations.get("chains", {})
            if not isinstance(chains, dict):
                return None
            for source_context, context_penalty in context_options:
                row = chains.get(source_context, {})
                if not isinstance(row, dict):
                    continue
                for signature, raw in row.items():
                    if not isinstance(raw, dict):
                        continue
                    connections = int(raw.get("connections", 0))
                    patterns = raw.get("input_patterns", {})
                    if not isinstance(patterns, dict) or not patterns:
                        continue
                    pattern, support = max(
                        (
                            (
                                str(value),
                                (
                                    int(count.get("legacy_support", 0))
                                    + int(count.get("trials", 0))
                                    if isinstance(count, dict)
                                    else int(count)
                                ),
                            )
                            for value, count in patterns.items()
                        ),
                        key=lambda item: item[1],
                    )
                    score = _human_demonstration_utility(
                        raw,
                        spirit=spirit,
                        support=support,
                        context_penalty=context_penalty,
                    )
                    if score is None:
                        continue
                    pattern_id = hashlib.sha256(pattern.encode("utf-8")).hexdigest()[:10]
                    viable.append(
                        (score, str(signature), pattern, pattern_id, connections)
                    )
        if not viable:
            return None
        score, signature, pattern, pattern_id, connections = max(viable)
        frames = _demonstration_frames(pattern, toward)
        if not frames:
            return None
        label = f"human:{signature}:{pattern_id}"
        self.queue.extend(frames)
        self.offense_outcomes.begin(
            label,
            context,
            frame=observation.frame,
            commitment=len(frames) + 18,
            enemy_hp=enemy.hp,
            me_hp=me.hp,
            spirit=spirit,
        )
        self.combo_target_hp = enemy.hp
        self.combo_confirm_deadline = observation.frame + min(24, len(frames))
        self.attack_cooldown = min(120, len(frames) + 24)
        self.counts[f"human_demo_attempts:{signature}:{pattern_id}"] += 1
        self.last_human_demonstration = {
            "frame": observation.frame,
            "context": context,
            "signature": signature,
            "pattern_id": pattern_id,
            "score": score,
            "connections": connections,
            "queued_frames": len(frames),
        }
        return self.queue.popleft(), f"human-punish:{signature}"

    def _safe_to_commit_skill(
        self,
        observation: PolicyObservation,
        label: str,
        distance: float,
        commitment: float | None = None,
    ) -> bool:
        """Reject casts whose startup/recovery window is already punishable."""
        me, enemy = observation.state.p1, observation.state.p2
        horizon = int(commitment or SKILL_COMMIT_FRAMES[label])

        # Starting a special while the opponent is already attacking is only
        # justified by a later, explicitly modelled invulnerability/graze case.
        if 300 <= enemy.action_id < 700:
            self.counts[f"commit_reject_enemy_action:{label}"] += 1
            return False
        closing_speed = max(0.0, abs(enemy.velocity_x) + abs(me.velocity_x))
        if distance - closing_speed * min(horizon, 18) < 150.0:
            self.counts[f"commit_reject_melee:{label}"] += 1
            return False
        if not observation.enemy_projectiles:
            return True

        hazards = self._hazards(observation.enemy_projectiles)
        result = self._evaluate_hazard(
            me.x,
            me.y,
            18.0,
            42.0,
            hazards,
            (_movement_candidate(0.0, 0.0),),
            horizon=min(horizon, 45),
            collision_margin=5.0,
        )[0]
        if not result.safe:
            self.counts[f"commit_reject_projectile:{label}"] += 1
            return False
        return True

    def _skill_candidate(self, label: str) -> ActionCandidate:
        action_id = SKILL_ACTIONS[label]
        profile = self.own_action_model.profiles.get(action_id)
        startup = SKILL_FALLBACK_STARTUP[label]
        commitment = float(SKILL_COMMIT_FRAMES[label])
        if profile is not None:
            if profile.projectile_samples:
                startup = max(1.0, profile.first_projectile_frame)
            if profile.completions:
                commitment = max(startup + 1.0, profile.total_frames)
        role = SKILL_INPUT[label][2]
        samples = self.counts.get(f"skill_duration_samples:{label}", 0)
        punished = self.counts.get(f"skill_punished:{label}", 0)
        punish_rate = punished / samples if samples else 0.0
        try:
            return ActionCandidate(
                label, startup, commitment, SKILL_DAMAGE[label], 200,
                250.0, 920.0, role, punish_rate,
            )
        except TypeError:
            return ActionCandidate(
                label, startup, commitment, SKILL_DAMAGE[label], 200,
                250.0, 920.0, role,
            )

    def decide(self, observation: PolicyObservation) -> PolicyDecision:
        state = observation.state
        me, enemy = state.p1, state.p2
        # Guard/motion direction must follow the fighter's native facing. An
        # airborne cross-up can pass x before the engine flips facing; using
        # raw positions at that instant presses forward and drops the guard.
        toward = (
            "right" if me.facing > 0 else "left"
            if me.facing < 0 else
            ("right" if me.x < enemy.x else "left")
        )
        back = "left" if toward == "right" else "right"
        distance = abs(enemy.x - me.x)
        spirit = int(getattr(me, "spirit", 1000))
        own_projectiles = tuple(getattr(observation, "own_projectiles", ()))
        self.last_own_projectile_count = len(own_projectiles)
        if not self.opponent_knowledge_seeded:
            prior = getattr(observation, "prior_opponent_model", {})
            seed = getattr(self.opponent, "seed", None)
            if prior and callable(seed):
                seed(prior)
                self.counts["seeded_opponent_actions"] += len(prior)
            self.opponent_knowledge_seeded = True
        if not self.projectile_knowledge_seeded:
            self.projectile_envelopes.import_state(
                getattr(observation, "prior_projectile_model", {}) or {}
            )
            self.projectile_knowledge_seeded = True
        if not self.defense_knowledge_seeded:
            self.defense_responses.import_state(
                getattr(observation, "prior_defense_model", {}) or {}
            )
            self.defense_knowledge_seeded = True
        if not self.offense_knowledge_seeded:
            self.offense_outcomes.import_state(
                getattr(observation, "prior_offense_model", {}) or {}
            )
            self.offense_knowledge_seeded = True
        if not self.human_knowledge_seeded:
            prior_human = getattr(observation, "prior_human_demonstrations", {})
            self.human_demonstrations = (
                prior_human if isinstance(prior_human, dict) else {}
            )
            self.human_knowledge_seeded = True
        if (
            observation.previous_state is not None
            and observation.previous_state.p1.hp <= 0 < me.hp
        ):
            _reset_model_episode(self.opponent)
            _reset_model_episode(self.own_action_model)
            self.queue.clear()
            self.evade_queue.clear()
            self.defense_responses.discard_episode()
            self.guard_chain_until = 0
            self.counts["round_model_resets"] += 1
        self.last_assessment = self.opponent.observe(
            observation.frame,
            enemy_action=enemy.action_id,
            me_hp=me.hp,
            projectile_count=len(observation.enemy_projectiles),
            active_hitbox=bool(getattr(enemy, "attack_boxes", ())),
        )
        self.own_action_model.observe(
            observation.frame,
            enemy_action=me.action_id,
            me_hp=enemy.hp,
            projectile_count=len(own_projectiles),
            active_hitbox=bool(getattr(me, "attack_boxes", ())),
        )
        self.offense_outcomes.observe(
            frame=observation.frame,
            enemy_hp=enemy.hp,
            me_hp=me.hp,
            spirit=spirit,
        )
        enemy_attacking = bool(getattr(enemy, "attack_boxes", ())) or self.last_assessment.phase in {
            "startup", "active", "unknown", "spell-danger"
        }
        enemy_attack_flags = int(getattr(enemy, "attack_flags", 0))
        me_in_hit_reaction = 50 <= me.action_id < 200
        previous_me = observation.previous_state.p1 if observation.previous_state else None
        if me_in_hit_reaction:
            # Block/hit stun is proof that the opponent's pressure is live.
            # Keep guard armed through short neutral gaps in multi-hit strings.
            self.guard_chain_until = max(self.guard_chain_until, observation.frame + 12)
        if (
            previous_me is not None
            and me.hp >= previous_me.hp
            and spirit < int(getattr(previous_me, "spirit", spirit))
        ):
            self.guard_chain_until = max(self.guard_chain_until, observation.frame + 20)
            self.counts["guard_contacts"] += 1
        confirmed_punish = 50 <= enemy.action_id < 200
        soft_recovery_probe = (
            self.last_assessment.phase == "recovery"
            and self.last_assessment.confidence >= 0.75
            and self.last_assessment.punish_window >= 8.0
        )

        melee_signature = (
            f"{enemy.action_id}:{enemy.action_sequence}:g{enemy_attack_flags & 0x6}"
            if 300 <= enemy.action_id < 400 else None
        )
        active_defense = self.defense_responses.episode
        if active_defense is not None and active_defense.signature != melee_signature:
            self.defense_responses.finish()
            active_defense = None
            self.defense_motion_issued = False
        if melee_signature is not None and active_defense is None:
            grounded_for_defense = me.y < 8.0
            can_retreat = (
                me.x > 105.0 if back == "left" else me.x < 1175.0
            )
            if grounded_for_defense:
                defense_candidates = ["high_guard", "low_guard"]
                if can_retreat:
                    defense_candidates.append("backdash")
                defense_candidates.append("jump_back")
            else:
                defense_candidates = ["high_guard"]
                if can_retreat:
                    defense_candidates.append("airdash_back")
                if spirit >= 250:
                    defense_candidates.append("flight_up")
            response = self.defense_responses.choose(
                melee_signature,
                tuple(defense_candidates),
            )
            prior_me = (
                observation.previous_state.p1
                if observation.previous_state is not None else me
            )
            self.defense_responses.begin(
                melee_signature,
                response,
                hp=prior_me.hp,
                spirit=int(getattr(prior_me, "spirit", spirit)),
                distance=distance,
            )
            self.evade_queue.clear()
            self.defense_motion_issued = False
        self.defense_responses.observe(
            hp=me.hp,
            spirit=spirit,
            distance=distance,
        )

        self.actions[me.action_id] += 1
        self.enemy_actions[enemy.action_id] += 1
        if self.last_me_action != me.action_id:
            self.counts["own_action_transitions"] += 1
            self.last_me_action = me.action_id
        if self.last_enemy_action != enemy.action_id:
            self.counts["enemy_action_transitions"] += 1
            self.last_enemy_action = enemy.action_id
        if self.last_enemy_hp is not None and enemy.hp < self.last_enemy_hp:
            self.counts["hit_confirms"] += 1
        self.last_enemy_hp = enemy.hp

        hurt_envelope = boxes_world_envelope(
            tuple(getattr(me, "hurt_boxes", ())),
            x=me.x,
            y=me.y,
            facing=me.facing,
        )
        if hurt_envelope is None:
            player_x, player_y = me.x, me.y + 42.0
            player_half_width, player_half_height = 18.0, 42.0
        else:
            left, bottom, right, top = hurt_envelope
            player_x, player_y = (left + right) * 0.5, (bottom + top) * 0.5
            player_half_width = (right - left) * 0.5
            player_half_height = (top - bottom) * 0.5

        took_damage = self.last_me_hp is not None and me.hp < self.last_me_hp
        previous_me_action = (
            observation.previous_state.p1.action_id
            if observation.previous_state is not None else None
        )
        learned_impact = self.projectile_envelopes.observe(
            observation.enemy_projectiles,
            player_x=player_x,
            player_y=player_y,
            player_half_width=player_half_width,
            player_half_height=player_half_height,
            took_damage=took_damage,
            first_contact=(
                observation.frame - self.last_damage_frame > 24
                or previous_me_action is None
                or not 50 <= previous_me_action < 200
            ),
        )

        if took_damage:
            damage = self.last_me_hp - me.hp
            nearest_damage_projectile = min(
                observation.enemy_projectiles,
                key=lambda projectile: math.hypot(projectile.x - me.x, projectile.y - me.y),
                default=None,
            )
            projectile_distance = (
                math.hypot(
                    nearest_damage_projectile.x - me.x,
                    nearest_damage_projectile.y - me.y,
                )
                if nearest_damage_projectile is not None else math.inf
            )
            self.counts["incoming_damage_events"] += 1
            self.counts["incoming_damage_total"] += damage
            if learned_impact is not None:
                self.counts[f"learned_projectile:{learned_impact.action_id}"] += 1
                self.counts[f"incoming_projectile:{learned_impact.action_id}"] += damage
            elif nearest_damage_projectile is not None and projectile_distance < 120.0:
                self.counts[
                    f"incoming_projectile:{nearest_damage_projectile.action_id}"
                ] += damage
            else:
                self.counts[f"incoming_action:{enemy.action_id}"] += damage
            self.counts[f"incoming_reaction:{me.action_id}"] += damage
            self.counts[f"incoming_after_intent:{self.previous_intent}"] += damage
            self.last_damage_context = {
                "frame": observation.frame,
                "damage": damage,
                "prior_intent": self.previous_intent,
                "enemy_action": enemy.action_id,
                "me_action": me.action_id,
                "projectile_action": (
                    learned_impact.action_id if learned_impact is not None else None
                ),
            }
            self.last_damage_frame = observation.frame

        if self.active_skill_label is not None:
            expected = SKILL_ACTIONS[self.active_skill_label]
            if me.action_id != expected:
                duration = observation.frame - self.active_skill_start_frame
                label = self.active_skill_label
                self.counts[f"skill_duration_frames:{label}"] += max(0, duration)
                self.counts[f"skill_duration_samples:{label}"] += 1
                if me.hp < self.active_skill_start_hp:
                    self.counts[f"skill_punished:{label}"] += 1
                self.active_skill_label = None
        self.last_me_hp = me.hp

        if self.combo_confirm_deadline:
            if self.combo_target_hp is not None and enemy.hp < self.combo_target_hp:
                self.counts["combo_hit_confirms"] += 1
                self.combo_confirm_deadline = 0
            elif observation.frame >= self.combo_confirm_deadline:
                self.queue.clear()
                self.counts["combo_aborted_no_hit"] += 1
                self.combo_confirm_deadline = 0

        projectile_threat = False
        airborne_threat = False
        nearest = math.inf
        evade_choice: str | None = None
        hazards = self._hazards(observation.enemy_projectiles)
        for projectile in hazards:
            dx = projectile.x - player_x
            dy = projectile.y - player_y
            nearest = min(nearest, math.hypot(dx, dy))
            moving_toward = dx * projectile.velocity_x < 0 or abs(projectile.velocity_x) < 0.2
            if abs(dx) < 260 and abs(dy) < 125 and moving_toward:
                projectile_threat = True
                airborne_threat |= projectile.y > me.y + 35

        if hazards:
            direction = 1.0 if toward == "right" else -1.0
            grounded = me.y < 8.0
            if grounded:
                labeled_candidates = (
                    ("stay", _movement_candidate(0.0, 0.0)),
                    (
                        "crouch",
                        _movement_candidate(
                            0.0,
                            0.0,
                            half_width=player_half_width,
                            half_height=24.0,
                            startup_frames=1,
                            center_offset_y=(me.y + 24.0) - player_y,
                        ),
                    ),
                    ("backdash", _movement_candidate(-direction * 13.0, 0.0, graze_frames=8, startup_frames=4)),
                    ("forward-dash", _movement_candidate(direction * 13.0, 0.0, graze_frames=8, startup_frames=4)),
                    ("jump", _movement_candidate(0.0, 11.0, startup_frames=1)),
                    ("superjump-back", _movement_candidate(-direction * 7.0, 13.0, graze_frames=8, startup_frames=3)),
                    ("superjump-forward", _movement_candidate(direction * 7.0, 13.0, graze_frames=8, startup_frames=3)),
                )
            else:
                labeled_candidates = (
                    ("stay", _movement_candidate(0.0, 0.0)),
                    ("airdash-back", _movement_candidate(-direction * 14.0, 0.0, graze_frames=7, startup_frames=4)),
                    ("airdash-forward", _movement_candidate(direction * 14.0, 0.0, graze_frames=7, startup_frames=4)),
                    ("flight-back", _movement_candidate(-direction * 10.0, 0.0, graze_frames=10, startup_frames=1)),
                    ("flight-up", _movement_candidate(0.0, 9.0, graze_frames=10, startup_frames=1)),
                    ("flight-down", _movement_candidate(0.0, -10.0, graze_frames=10, startup_frames=1)),
                )
            labels = tuple(label for label, _candidate in labeled_candidates)
            candidates = tuple(candidate for _label, candidate in labeled_candidates)
            risk_horizon = 30
            risk = self._evaluate_hazard(
                player_x,
                player_y,
                player_half_width,
                player_half_height,
                hazards,
                candidates,
                horizon=risk_horizon,
                collision_margin=5.0,
            )
            eligible: list[tuple[float, str]] = []
            for label, candidate, result in zip(labels, candidates, risk):
                if label == "stay" or not result.safe or me.action_id >= 50:
                    continue
                final_x = getattr(
                    result, "final_x", me.x + candidate.velocity_x * risk_horizon
                )
                if not 45.0 <= final_x <= 1235.0:
                    continue
                if label.startswith("flight") and spirit < 400:
                    continue
                if label.endswith("forward") or label == "forward-dash":
                    if enemy_attacking and abs(enemy.x - final_x) < 100.0:
                        continue
                # Clearance dominates; a small cost keeps crouch/jump ahead of
                # spirit-consuming flight when both are equally safe.
                cost = 18.0 if label.startswith("flight") else 0.0
                eligible.append((result.minimum_clearance - cost, label))
            if eligible:
                evade_choice = max(eligible)[1]
            self.last_risk = {
                "backend": self.hazard.backend,
                "native_attack_boxes": len(hazards),
                "player_half_width": player_half_width,
                "player_half_height": player_half_height,
                "best": evade_choice or "none",
                **{
                    f"{label}_clearance": result.minimum_clearance
                    for label, result in zip(labels, risk)
                },
                **{f"{label}_safe": result.safe for label, result in zip(labels, risk)},
            }
            # A projectile that is inside the coarse proximity window but has
            # no collision on the stationary 10-frame path should not force a
            # long guard. This removes false positives from receding knives.
            if not risk[0].safe:
                projectile_threat = True
            elif risk[0].minimum_clearance > 8.0:
                projectile_threat = False

        if self.pending_skill > 0:
            expected_action = SKILL_ACTIONS.get(self.pending_skill_label or "")
            if expected_action is not None and me.action_id == expected_action:
                self.counts["skill_successes"] += 1
                self.counts[f"skill_success:{self.pending_skill_label}"] += 1
                self.pending_skill = 0
                self.pending_skill_label = None
                self.queue.clear()
                self.active_skill_label = next(
                    label for label, action in SKILL_ACTIONS.items()
                    if action == expected_action
                )
                self.active_skill_start_frame = observation.frame
                self.active_skill_start_hp = me.hp
            else:
                self.pending_skill -= 1
                if self.pending_skill == 0:
                    self.counts[f"skill_failed:{self.pending_skill_label}"] += 1
                    self.pending_skill_label = None

        if me_in_hit_reaction:
            self.queue.clear()
            self.evade_queue.clear()
            self.combo_confirm_deadline = 0
            keys = {back}
            intent = "recover-guard"
        elif 500 <= me.action_id < 600:
            # A special has already committed.  Do not enqueue imaginary
            # dodges during its startup/recovery; buffer guard only when a
            # threat develops and learn whether the cast was punished.
            self.evade_queue.clear()
            keys = {back} if projectile_threat or enemy_attacking else set()
            intent = "skill-commit-guard" if keys else "skill-commit"
        elif 300 <= enemy.action_id < 400 and (
            enemy_attacking or distance < 280.0
        ):
            # Defense-first: a strike startup invalidates any not-yet-active
            # motion or dodge macro. Facing-based guard takes priority over all
            # queued offense and prevents the first hit of a full chain.
            self.queue.clear()
            self.combo_confirm_deadline = 0
            response = (
                self.defense_responses.episode.response
                if self.defense_responses.episode is not None else "high_guard"
            )
            response = _native_guard_response(enemy_attack_flags, response)
            mark_applied = getattr(self.defense_responses, "mark_applied", None)
            if callable(mark_applied):
                mark_applied()
            if response in {"backdash", "airdash_back"} and me.action_id < 50:
                if not self.defense_motion_issued:
                    self.evade_queue.extend(_dash_frames(back, hold_frames=8))
                    self.defense_motion_issued = True
                keys = self.evade_queue.popleft() if self.evade_queue else {back}
                intent = f"learned-defense-{response}"
            elif response == "jump_back" and me.action_id < 50:
                if not self.defense_motion_issued:
                    self.evade_queue.extend(
                        _jump_frames(toward, "back", hold_frames=8)
                    )
                    self.defense_motion_issued = True
                keys = self.evade_queue.popleft() if self.evade_queue else {back}
                intent = "learned-defense-jump-back"
            elif response == "flight_up" and me.action_id < 50:
                if not self.defense_motion_issued:
                    self.evade_queue.extend(_flight_frames({"up"}, hold_frames=8))
                    self.defense_motion_issued = True
                keys = self.evade_queue.popleft() if self.evade_queue else {back}
                intent = "learned-defense-flight-up"
            else:
                self.evade_queue.clear()
                low = response == "low_guard"
                keys = {back, "down"} if low else {back}
                intent = "strike-guard-low" if low else "strike-guard-high"
        elif observation.frame <= self.guard_chain_until and (
            distance < 280.0 or hazards or enemy_attacking
        ):
            # Do not release back between hits. Many strings briefly return to
            # a non-active pose while the next hit is already buffered.
            self.queue.clear()
            response = (
                self.defense_responses.episode.response
                if self.defense_responses.episode is not None else "high_guard"
            )
            response = _native_guard_response(enemy_attack_flags, response)
            low = response == "low_guard"
            keys = {back, "down"} if low else {back}
            intent = "guard-chain-low" if low else "guard-chain-high"
        elif self.evade_queue:
            keys = self.evade_queue.popleft()
            intent = "evade-sequence"
        elif projectile_threat:
            self.queue.clear()
            self.combo_confirm_deadline = 0
            self.counts["projectile_guards"] += 1
            # 623B is projectile-grazing anti-air, but it is not strike
            # invulnerable. Only attempt it against a high projectile with no
            # nearby active enemy; otherwise keep the robust guard fallback.
            if (
                evade_choice is not None
                and self.evade_cooldown == 0
                and distance > 190
                and me.action_id < 50
            ):
                physical_back = "left" if toward == "right" else "right"
                if evade_choice == "crouch":
                    keys = {"down"}
                elif evade_choice in {"backdash", "airdash-back"}:
                    self.evade_queue.extend(_dash_frames(physical_back))
                    keys = self.evade_queue.popleft()
                elif evade_choice in {"forward-dash", "airdash-forward"}:
                    self.evade_queue.extend(_dash_frames(toward))
                    keys = self.evade_queue.popleft()
                elif evade_choice == "jump":
                    self.evade_queue.extend(_jump_frames(toward))
                    keys = self.evade_queue.popleft()
                elif evade_choice == "superjump-back":
                    self.evade_queue.extend(
                        _jump_frames(toward, "back", super_jump=True)
                    )
                    keys = self.evade_queue.popleft()
                elif evade_choice == "superjump-forward":
                    self.evade_queue.extend(
                        _jump_frames(toward, "toward", super_jump=True)
                    )
                    keys = self.evade_queue.popleft()
                elif evade_choice == "flight-back":
                    self.evade_queue.extend(_flight_frames({physical_back}))
                    keys = self.evade_queue.popleft()
                elif evade_choice == "flight-up":
                    self.evade_queue.extend(_flight_frames({"up"}))
                    keys = self.evade_queue.popleft()
                elif evade_choice == "flight-down":
                    self.evade_queue.extend(_flight_frames({"down"}))
                    keys = self.evade_queue.popleft()
                else:
                    keys = {back}
                self.evade_cooldown = 30
                self.counts[f"evade:{evade_choice}"] += 1
                intent = f"projectile-evade:{evade_choice}"
            else:
                keys = {back} if enemy.y >= 36 else {back, "down"}
                intent = "projectile-guard"
        elif self.last_assessment.spell:
            # We do not spend S/D spell cards yet, but enemy spell actions are
            # always modelled as high commitment danger.  Wait for native
            # projectile geometry instead of trying to steal an early turn.
            self.queue.clear()
            keys = {back} if me.y < 8.0 else {back, "a"}
            intent = "enemy-spell-defend"
        elif enemy_attacking:
            # Ranged normals/skills are guarded from startup, before their
            # projectile object necessarily exists. Geometry takes over on the
            # first frame a projectile is observable.
            self.queue.clear()
            self.combo_confirm_deadline = 0
            keys = {back}
            intent = "ranged-startup-guard"
        elif self.queue:
            keys = self.queue.popleft()
            intent = "sequence"
        elif confirmed_punish and distance < 150 and me.action_id < 50:
            selected = self._start_human_demonstration(observation, toward)
            if selected is None:
                selected = self._start_combo(observation, toward, punish=True)
            if selected is None:
                keys, intent = {back}, "punish-unavailable"
            else:
                keys, intent = selected
        elif (
            soft_recovery_probe
            and distance < 58.0
            and me.action_id < 50
            and self.attack_cooldown == 0
            and not hazards
        ):
            # A learned recovery estimate is not as strong as native hitstun.
            # Probe with one close A; never commit a full route on timing alone.
            keys = {"z"}
            intent = "soft-punish-probe"
            self.attack_cooldown = 18
            self.counts["soft_punish_probes"] += 1
        elif distance > 310:
            if spirit < 400:
                keys = {back}
                intent = "spirit-recover"
            elif (
                self.skill_cooldown == 0
                and me.action_id < 50
                and self.last_assessment.phase in {"reaction", "recovery"}
            ):
                # Guide-derived roles: 236 is fast head-on control, 214 is a
                # persistent bounce wall useful before the opponent advances.
                closing = (enemy.x - me.x) * enemy.velocity_x < 0
                # Six frames for an Easy CPU decision plus a conservative
                # twelve units/frame ground approach after that.
                enemy_time_to_contact = (
                    6.0 + max(0.0, distance - 105.0) / 12.0
                    if closing else float("inf")
                )
                try:
                    tactical_context = TacticalContext(
                        distance=distance,
                        vertical_gap=enemy.y - me.y,
                        spirit=spirit,
                        enemy_phase=self.last_assessment.phase,
                        punish_window=self.last_assessment.punish_window,
                        own_projectiles=len(own_projectiles),
                        enemy_closing=closing,
                        enemy_time_to_contact=enemy_time_to_contact,
                    )
                except TypeError:
                    tactical_context = TacticalContext(
                        distance=distance,
                        vertical_gap=enemy.y - me.y,
                        spirit=spirit,
                        enemy_phase=self.last_assessment.phase,
                        punish_window=self.last_assessment.punish_window,
                        own_projectiles=len(own_projectiles),
                        enemy_closing=closing,
                    )
                ranked = rank_actions(
                    tuple(self._skill_candidate(label) for label in SKILL_INPUT),
                    tactical_context,
                )
                self.last_tactical_scores = [
                    {
                        "name": item.candidate.name,
                        "score": item.score,
                        "hit_probability": item.hit_probability,
                        "reason": item.reason,
                        "startup": item.candidate.startup,
                        "commitment": item.candidate.commitment,
                    }
                    for item in ranked
                ]
                chosen = next(
                    (
                        item for item in ranked
                        if item.reason == "ok"
                        and (
                            self.counts.get(
                                f"skill_punished:{item.candidate.name}", 0
                            )
                            * 2
                            < max(
                                1,
                                self.counts.get(
                                    f"skill_duration_samples:{item.candidate.name}", 0
                                ),
                            )
                            or self.last_assessment.phase in {"reaction", "recovery"}
                        )
                    ),
                    None,
                )
                if chosen is None:
                    keys = {back}
                    intent = "no-safe-skill-candidate"
                else:
                    label = chosen.candidate.name
                    motion, button, _role = SKILL_INPUT[label]
                if chosen is not None and self._safe_to_commit_skill(
                    observation, label, distance, chosen.candidate.commitment
                ):
                    self.skill_index += 1
                    self.queue.extend(build_motion_frames(motion, button, toward))
                    self.skill_cooldown = 105
                    self.pending_skill = 35
                    self.pending_skill_label = label
                    self.counts[f"skill_attempts:{label}"] += 1
                    self.offense_outcomes.begin(
                        label,
                        combat_context(
                            distance=distance,
                            enemy_y=enemy.y,
                            enemy_x=enemy.x,
                            phase=self.last_assessment.phase,
                        ),
                        frame=observation.frame,
                        commitment=int(chosen.candidate.commitment) + 12,
                        enemy_hp=enemy.hp,
                        me_hp=me.hp,
                        spirit=spirit,
                    )
                    keys = self.queue.popleft()
                    intent = f"space-control-{label}"
                elif chosen is not None:
                    keys = {back}
                    intent = f"unsafe-skill-reject-{label}"
            else:
                keys = set()
                intent = "far-neutral-watch"
        elif distance > 92:
            if spirit < 300:
                keys = {back}
                intent = "spirit-recover"
            elif self.last_assessment.phase in {"reaction", "recovery"}:
                keys = {toward}
                intent = "close-during-advantage"
            else:
                keys = {back} if distance < 185.0 else set()
                intent = "neutral-spacing" if keys else "neutral-watch"
        elif (
            self.attack_cooldown == 0
            and me.action_id < 50
            and self.last_assessment.phase in {"reaction", "recovery"}
        ):
            selected = self._start_combo(observation, toward, punish=False)
            if selected is None:
                keys, intent = {toward}, "close-for-combo"
            else:
                keys, intent = selected
        else:
            keys = {back}
            intent = "close-neutral-guard"

        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        self.skill_cooldown = max(0, self.skill_cooldown - 1)
        self.evade_cooldown = max(0, self.evade_cooldown - 1)
        return self._decision(keys, intent)

    def metrics(self) -> dict[str, object]:
        human_contexts = self.human_demonstrations.get("contexts", {})
        if not isinstance(human_contexts, dict):
            human_contexts = {}
        return {
            "counts": dict(self.counts),
            "own_action_frames": {str(k): v for k, v in self.actions.most_common()},
            "enemy_action_frames": {str(k): v for k, v in self.enemy_actions.most_common()},
            "queued_frames": len(self.queue),
            "evade_queued_frames": len(self.evade_queue),
            "hazard_backend": self.hazard.backend,
            "last_risk": self.last_risk,
            "observed_own_projectiles": self.last_own_projectile_count,
            "projectile_envelopes": self.projectile_envelopes.metrics(),
            "projectile_envelope_state": self.projectile_envelopes.export_state(),
            "last_damage_context": self.last_damage_context,
            "defense_responses": self.defense_responses.metrics(),
            "defense_response_state": self.defense_responses.export_state(),
            "offense_outcomes": self.offense_outcomes.metrics(),
            "offense_outcome_state": self.offense_outcomes.export_state(),
            "human_demonstrations": {
                "contexts": len(human_contexts),
                "candidates": sum(
                    len(candidates)
                    for candidates in human_contexts.values()
                    if isinstance(candidates, list)
                ),
                "attempts": sum(
                    count
                    for name, count in self.counts.items()
                    if name.startswith("human_demo_attempts:")
                ),
                "last_selected": self.last_human_demonstration or None,
            },
            "opponent_assessment": {
                "action": self.last_assessment.action_id,
                "elapsed": self.last_assessment.elapsed,
                "phase": self.last_assessment.phase,
                "remaining": self.last_assessment.remaining_frames,
                "punish_window": self.last_assessment.punish_window,
                "confidence": self.last_assessment.confidence,
                "spell": self.last_assessment.spell,
            },
            "opponent_model": self.opponent.metrics(),
            "own_action_model": self.own_action_model.metrics(),
            "tactical_scores": self.last_tactical_scores,
            "performance": {
                "hazard_calls": self.hazard_calls,
                "hazard_average_us": (
                    self.hazard_total_ns / self.hazard_calls / 1000.0
                    if self.hazard_calls else 0.0
                ),
                "hazard_max_us": self.hazard_max_ns / 1000.0,
            },
        }


def create_policy() -> SakuyaAdaptivePolicy:
    return SakuyaAdaptivePolicy()
