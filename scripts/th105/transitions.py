"""Bounded event-aligned combat transitions for future offline learners."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .reward import basis_points


TRANSITION_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 1
ACTION_SCHEMA_VERSION = 2


class TransitionWriter(Protocol):
    def write(self, record: dict[str, object]) -> None: ...


def _quantize(value: object, scale: float = 1.0, limit: int = 32767) -> int:
    try:
        result = round(float(value) / scale)
    except (TypeError, ValueError):
        return 0
    return max(-limit, min(limit, result))


def _fighter_features(fighter: Any) -> dict[str, object]:
    return {
        "character_vtable": f"0x{int(getattr(fighter, 'vtable', 0)):08X}",
        "x_q4": _quantize(getattr(fighter, "x", 0.0), 4.0),
        "y_q4": _quantize(getattr(fighter, "y", 0.0), 4.0),
        "velocity_x_q4": _quantize(getattr(fighter, "velocity_x", 0.0), 0.25),
        "velocity_y_q4": _quantize(getattr(fighter, "velocity_y", 0.0), 0.25),
        "facing": int(getattr(fighter, "facing", 0)),
        "action": int(getattr(fighter, "action_id", 0)),
        "sequence": int(getattr(fighter, "action_sequence", 0)),
        "pose": int(getattr(fighter, "action_pose", 0)),
        "action_frame": int(getattr(fighter, "action_frame", 0)),
        "frame_data_flags": int(getattr(fighter, "frame_data_flags", 0)),
        "attack_flags": int(getattr(fighter, "attack_flags", 0)),
        "hp": int(getattr(fighter, "hp", 0)),
        "max_hp": max(1, int(getattr(fighter, "max_hp", 10000))),
        "hp_bp": basis_points(
            int(getattr(fighter, "hp", 0)),
            max(1, int(getattr(fighter, "max_hp", 10000))),
        ),
        "spirit": int(getattr(fighter, "spirit", 0)),
        "max_spirit": 1000,
        "spirit_bp": basis_points(int(getattr(fighter, "spirit", 0)), 1000),
        "collision_state": int(getattr(fighter, "collision_state", 0)),
        "hurt_box_count": len(tuple(getattr(fighter, "hurt_boxes", ()))),
        "attack_box_count": len(tuple(getattr(fighter, "attack_boxes", ()))),
    }


def _projectile_features(projectile: Any, me: Any) -> dict[str, int]:
    return {
        "action": int(getattr(projectile, "action_id", 0)),
        "relative_x_q4": _quantize(
            float(getattr(projectile, "x", 0.0)) - float(getattr(me, "x", 0.0)),
            4.0,
        ),
        "relative_y_q4": _quantize(
            float(getattr(projectile, "y", 0.0)) - float(getattr(me, "y", 0.0)),
            4.0,
        ),
        "velocity_x_q4": _quantize(getattr(projectile, "velocity_x", 0.0), 0.25),
        "velocity_y_q4": _quantize(getattr(projectile, "velocity_y", 0.0), 0.25),
        "acceleration_x_q64": _quantize(
            getattr(projectile, "acceleration_x", 0.0), 1.0 / 64.0
        ),
        "acceleration_y_q64": _quantize(
            getattr(projectile, "acceleration_y", 0.0), 1.0 / 64.0
        ),
        "facing": int(getattr(projectile, "facing", 0)),
        "attack_flags": int(getattr(projectile, "attack_flags", 0)),
    }


def transition_state(
    state: Any,
    enemy_projectiles: tuple[Any, ...] = (),
    own_projectiles: tuple[Any, ...] = (),
) -> dict[str, object]:
    me, enemy = state.p1, state.p2
    relative_x = float(getattr(enemy, "x", 0.0)) - float(getattr(me, "x", 0.0))
    relative_y = float(getattr(enemy, "y", 0.0)) - float(getattr(me, "y", 0.0))
    closing_speed = (
        float(getattr(enemy, "velocity_x", 0.0))
        - float(getattr(me, "velocity_x", 0.0))
    ) * (1.0 if relative_x >= 0.0 else -1.0)
    return {
        "relative_x_q4": _quantize(relative_x, 4.0),
        "relative_y_q4": _quantize(relative_y, 4.0),
        "closing_speed_q4": _quantize(closing_speed, 0.25),
        "self": _fighter_features(me),
        "enemy": _fighter_features(enemy),
        # Retain exact observed object facts, but cap pathological/corrupt lists.
        "enemy_projectiles": [
            _projectile_features(projectile, me)
            for projectile in enemy_projectiles[:32]
        ],
        "own_projectiles": [
            _projectile_features(projectile, me)
            for projectile in own_projectiles[:32]
        ],
    }


@dataclass
class _ActiveOption:
    episode_id: str
    step: int
    start_frame: int
    action: str
    legal_actions: tuple[str, ...] | None
    behavior_probability: float
    start_state: dict[str, object]
    start_enemy_hp: int
    start_me_hp: int
    start_spirit: int
    max_enemy_hp: int
    max_me_hp: int
    max_spirit: int
    policy_generation: int
    policy_sha256: str | None
    key_trace_rle: list[dict[str, object]] = field(default_factory=list)


class OptionTransitionRecorder:
    """Segment decisions on intent, legal-set, or policy-generation changes."""

    def __init__(
        self,
        writer: TransitionWriter,
        *,
        session_id: str,
        opponent: str,
        difficulty: str,
        game_build_sha256: str | None = None,
        offline_policy_sha256: str | None = None,
    ) -> None:
        self.writer = writer
        self.session_id = session_id
        self.opponent = opponent
        self.difficulty = difficulty
        self.game_build_sha256 = game_build_sha256
        self.offline_policy_sha256 = offline_policy_sha256
        self.episode_id: str | None = None
        self.step = 0
        self.active: _ActiveOption | None = None
        self.transitions = 0
        self._pending: list[dict[str, object]] = []

    def _emit(self, record: dict[str, object]) -> None:
        bulk_write = getattr(self.writer, "write_many", None)
        if not callable(bulk_write):
            self.writer.write(record)
            return
        self._pending.append(record)
        if len(self._pending) >= 32:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        bulk_write = getattr(self.writer, "write_many", None)
        if callable(bulk_write):
            bulk_write(pending)
        else:
            for record in pending:
                self.writer.write(record)

    def _new_episode(self) -> None:
        self.episode_id = uuid.uuid4().hex
        self.step = 0

    @staticmethod
    def _keys_key(keys: object) -> str:
        return "+".join(sorted(str(key) for key in keys))

    @staticmethod
    def _legal_actions(decision: Any) -> tuple[str, ...] | None:
        raw = getattr(decision, "legal_actions", None)
        if not isinstance(raw, (tuple, list)):
            return None
        # Canonical ordering makes this both compact under gzip and stable as
        # an option-boundary key.
        return tuple(sorted({str(action) for action in raw}))

    def _append_keys(self, keys: object) -> None:
        if self.active is None:
            return
        key = self._keys_key(keys)
        if self.active.key_trace_rle and self.active.key_trace_rle[-1]["keys"] == key:
            self.active.key_trace_rle[-1]["frames"] = int(
                self.active.key_trace_rle[-1]["frames"]
            ) + 1
        else:
            self.active.key_trace_rle.append({"keys": key, "frames": 1})

    def _start(
        self,
        *,
        frame: int,
        state: Any,
        decision: Any,
        enemy_projectiles: tuple[Any, ...],
        own_projectiles: tuple[Any, ...],
        policy_generation: int,
        policy_sha256: str | None,
    ) -> None:
        if self.episode_id is None:
            self._new_episode()
        assert self.episode_id is not None
        me, enemy = state.p1, state.p2
        legal_actions = self._legal_actions(decision)
        probability = float(getattr(decision, "behavior_probability", 1.0))
        if not 0.0 < probability <= 1.0:
            probability = 1.0
        self.active = _ActiveOption(
            episode_id=self.episode_id,
            step=self.step,
            start_frame=frame,
            action=str(decision.intent),
            legal_actions=legal_actions,
            behavior_probability=probability,
            start_state=transition_state(state, enemy_projectiles, own_projectiles),
            start_enemy_hp=int(enemy.hp),
            start_me_hp=int(me.hp),
            start_spirit=int(me.spirit),
            max_enemy_hp=max(1, int(enemy.max_hp)),
            max_me_hp=max(1, int(me.max_hp)),
            max_spirit=1000,
            policy_generation=policy_generation,
            policy_sha256=policy_sha256,
        )
        self._append_keys(decision.keys)

    def observe(
        self,
        *,
        frame: int,
        state: Any,
        decision: Any,
        enemy_projectiles: tuple[Any, ...] = (),
        own_projectiles: tuple[Any, ...] = (),
        policy_generation: int = 0,
        policy_sha256: str | None = None,
    ) -> None:
        boundary = (
            self.active is not None
            and (
                self.active.action != str(decision.intent)
                or self.active.legal_actions != self._legal_actions(decision)
                or self.active.policy_generation != policy_generation
            )
        )
        if boundary:
            self._finish(
                frame=frame,
                state=state,
                enemy_projectiles=enemy_projectiles,
                own_projectiles=own_projectiles,
                terminal=None,
                truncated=False,
            )
        if self.active is None:
            self._start(
                frame=frame,
                state=state,
                decision=decision,
                enemy_projectiles=enemy_projectiles,
                own_projectiles=own_projectiles,
                policy_generation=policy_generation,
                policy_sha256=policy_sha256,
            )
        elif not boundary:
            self._append_keys(decision.keys)

    def _finish(
        self,
        *,
        frame: int,
        state: Any | None,
        enemy_projectiles: tuple[Any, ...],
        own_projectiles: tuple[Any, ...],
        terminal: str | None,
        truncated: bool,
    ) -> None:
        option = self.active
        self.active = None
        if option is None:
            return
        me = state.p1 if state is not None else None
        enemy = state.p2 if state is not None else None
        enemy_hp = int(getattr(enemy, "hp", option.start_enemy_hp))
        me_hp = int(getattr(me, "hp", option.start_me_hp))
        spirit = int(getattr(me, "spirit", option.start_spirit))
        damage = max(0, option.start_enemy_hp - enemy_hp)
        self_damage = max(0, option.start_me_hp - me_hp)
        spirit_delta = spirit - option.start_spirit
        self._emit(
            {
                "schema_version": TRANSITION_SCHEMA_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_id": option.episode_id,
                "transition_id": f"{option.episode_id}:{option.step:06d}",
                "step": option.step,
                "opponent": self.opponent,
                "difficulty": self.difficulty,
                "game_build_sha256": self.game_build_sha256,
                "offline_policy_sha256": self.offline_policy_sha256,
                "policy_generation": option.policy_generation,
                "policy_sha256": option.policy_sha256,
                "boundary": "intent-change-v1",
                "state": option.start_state,
                "legal_actions": (
                    list(option.legal_actions)
                    if option.legal_actions is not None else None
                ),
                "legal_actions_known": option.legal_actions is not None,
                "action": option.action,
                "behavior_probability": option.behavior_probability,
                "key_trace_rle": option.key_trace_rle,
                "duration_frames": max(1, frame - option.start_frame),
                "outcome": {
                    "damage": damage,
                    "damage_bp": basis_points(damage, option.max_enemy_hp),
                    "self_damage": self_damage,
                    "self_damage_bp": basis_points(
                        self_damage, option.max_me_hp
                    ),
                    "spirit_delta": spirit_delta,
                    "spirit_cost_bp": basis_points(
                        max(0, -spirit_delta), option.max_spirit
                    ),
                    "punished": self_damage > 0,
                    "terminal": terminal,
                    "truncated": truncated,
                },
                "next_state": (
                    transition_state(state, enemy_projectiles, own_projectiles)
                    if state is not None else None
                ),
            }
        )
        self.transitions += 1
        self.step += 1

    def finish_terminal(
        self,
        *,
        frame: int,
        state: Any,
        won: bool | None,
    ) -> None:
        terminal = "win" if won is True else "loss" if won is False else "draw"
        self._finish(
            frame=frame,
            state=state,
            enemy_projectiles=(),
            own_projectiles=(),
            terminal=terminal,
            truncated=False,
        )
        self.episode_id = None
        self.flush()

    def finish_truncated(self, *, frame: int, state: Any | None) -> None:
        self._finish(
            frame=frame,
            state=state,
            enemy_projectiles=(),
            own_projectiles=(),
            terminal=None,
            truncated=True,
        )
        self.episode_id = None
        self.flush()
