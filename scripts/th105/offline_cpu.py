"""Stable tabular features for CPU outcome models and policy distillation."""

from __future__ import annotations

from typing import Iterable

from .schema import (
    ACTION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    TRAINING_GENERATION,
    TRANSITION_SCHEMA_VERSION,
)


CPU_FEATURE_SCHEMA_VERSION = 1
CATEGORICAL_FEATURES = (
    "difficulty",
    "opponent",
    "option",
    "self_character",
    "enemy_character",
)
FIGHTER_NUMERIC = (
    "x_q4",
    "y_q4",
    "velocity_x_q4",
    "velocity_y_q4",
    "facing",
    "action",
    "sequence",
    "pose",
    "action_frame",
    "frame_data_flags",
    "attack_flags",
    "hp_bp",
    "spirit_bp",
    "collision_state",
    "hurt_box_count",
    "attack_box_count",
)
NUMERIC_FEATURES = (
    "relative_x_q4",
    "relative_y_q4",
    "closing_speed_q4",
    *(f"self_{name}" for name in FIGHTER_NUMERIC),
    *(f"enemy_{name}" for name in FIGHTER_NUMERIC),
    "enemy_projectile_count",
    "enemy_projectile_nearest_action",
    "enemy_projectile_nearest_x_q4",
    "enemy_projectile_nearest_y_q4",
    "enemy_projectile_nearest_vx_q4",
    "enemy_projectile_nearest_vy_q4",
    "enemy_projectile_nearest_attack_flags",
    "own_projectile_count",
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


def validate_transition_schemas(records: list[dict[str, object]]) -> None:
    """Reject mixed/legacy action spaces before any offline fitting."""
    expected = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    for name, version in expected.items():
        observed = {int(record.get(name, 0)) for record in records}
        if observed != {version}:
            raise ValueError(f"incompatible {name}: {sorted(observed)} != [{version}]")
    generations = {str(record.get("training_generation", "")) for record in records}
    if generations != {TRAINING_GENERATION}:
        raise ValueError(
            "incompatible training generations: "
            f"{sorted(generations)} != [{TRAINING_GENERATION!r}]"
        )


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _number(mapping: dict[str, object], name: str) -> float:
    try:
        return float(mapping.get(name, 0))
    except (TypeError, ValueError):
        return 0.0


def _nearest_projectile(projectiles: object) -> dict[str, object]:
    if not isinstance(projectiles, list) or not projectiles:
        return {}
    candidates = [item for item in projectiles if isinstance(item, dict)]
    if not candidates:
        return {}
    return min(
        candidates,
        key=lambda item: abs(_number(item, "relative_x_q4"))
        + abs(_number(item, "relative_y_q4")),
    )


def transition_features(record: dict[str, object]) -> dict[str, str | float]:
    state = _mapping(record.get("state"))
    me = _mapping(state.get("self"))
    enemy = _mapping(state.get("enemy"))
    enemy_projectiles = state.get("enemy_projectiles")
    own_projectiles = state.get("own_projectiles")
    nearest = _nearest_projectile(enemy_projectiles)
    features: dict[str, str | float] = {
        "difficulty": str(record.get("difficulty", "unknown")),
        "opponent": str(record.get("opponent", "unknown")),
        "option": str(record.get("action", "unknown")),
        "self_character": str(me.get("character_vtable", "unknown")),
        "enemy_character": str(enemy.get("character_vtable", "unknown")),
        "relative_x_q4": _number(state, "relative_x_q4"),
        "relative_y_q4": _number(state, "relative_y_q4"),
        "closing_speed_q4": _number(state, "closing_speed_q4"),
    }
    for prefix, fighter in (("self", me), ("enemy", enemy)):
        for name in FIGHTER_NUMERIC:
            features[f"{prefix}_{name}"] = _number(fighter, name)
    enemy_count = len(enemy_projectiles) if isinstance(enemy_projectiles, list) else 0
    own_count = len(own_projectiles) if isinstance(own_projectiles, list) else 0
    features.update(
        {
            "enemy_projectile_count": float(enemy_count),
            "enemy_projectile_nearest_action": _number(nearest, "action"),
            "enemy_projectile_nearest_x_q4": _number(nearest, "relative_x_q4"),
            "enemy_projectile_nearest_y_q4": _number(nearest, "relative_y_q4"),
            "enemy_projectile_nearest_vx_q4": _number(nearest, "velocity_x_q4"),
            "enemy_projectile_nearest_vy_q4": _number(nearest, "velocity_y_q4"),
            "enemy_projectile_nearest_attack_flags": _number(nearest, "attack_flags"),
            "own_projectile_count": float(own_count),
        }
    )
    return features


def feature_vector(record: dict[str, object]) -> list[str | float]:
    features = transition_features(record)
    return [features[name] for name in FEATURE_NAMES]


def candidate_prediction_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Expand factual rows with every logged native-gated candidate.

    The first ``len(records)`` rows remain factual and therefore preserve the
    chronological train/validation indices. Additional rows are used only for
    counterfactual model prediction and compact distillation, never as targets.
    """
    factual: list[dict[str, object]] = []
    counterfactual: list[dict[str, object]] = []
    for record in records:
        actual = str(record.get("action", "unknown"))
        base = dict(record)
        base["__factual_action"] = actual
        factual.append(base)
        raw_legal = record.get("legal_actions")
        legal = (
            sorted({str(action) for action in raw_legal})
            if isinstance(raw_legal, list)
            else []
        )
        for action in legal:
            if action == actual:
                continue
            candidate = dict(record)
            candidate["action"] = action
            candidate["__factual_action"] = actual
            counterfactual.append(candidate)
    return factual + counterfactual


def outcome_targets(record: dict[str, object]) -> dict[str, float]:
    outcome = _mapping(record.get("outcome"))
    terminal = str(outcome.get("terminal") or "").casefold()
    terminal_value = 1.0 if terminal == "win" else -1.0 if terminal == "loss" else 0.0
    return {
        "damage_bp": _number(outcome, "damage_bp"),
        "connection_probability": 1.0 if _number(outcome, "damage_bp") > 0 else 0.0,
        "self_damage_bp": _number(outcome, "self_damage_bp"),
        "spirit_cost_bp": _number(outcome, "spirit_cost_bp"),
        "punished_probability": 1.0 if bool(outcome.get("punished")) else 0.0,
        "commitment_frames": max(1.0, _number(record, "duration_frames")),
        "terminal_value": terminal_value,
    }


def distillation_context(record: dict[str, object]) -> str:
    state = _mapping(record.get("state"))
    me = _mapping(state.get("self"))
    enemy = _mapping(state.get("enemy"))
    relative_x = abs(_number(state, "relative_x_q4"))
    distance = "close" if relative_x < 20 else "mid" if relative_x < 65 else "far"
    altitude = "ground" if abs(_number(state, "relative_y_q4")) < 11 else "air"
    enemy_x = _number(enemy, "x_q4")
    corner = "corner" if enemy_x < 38 or enemy_x > 282 else "field"
    self_x = _number(me, "x_q4")
    self_zone = (
        "left-wall"
        if self_x <= 30
        else "right-wall"
        if self_x >= 290
        else "field"
    )
    projectiles = state.get("enemy_projectiles")
    projectile_count = len(projectiles) if isinstance(projectiles, list) else 0
    projectile_bucket = (
        "p0" if projectile_count == 0
        else "p1-3" if projectile_count <= 3
        else "p4-8" if projectile_count <= 8
        else "p9+"
    )
    self_hp = int(_number(me, "hp_bp") // 1000)
    enemy_hp = int(_number(enemy, "hp_bp") // 1000)
    return ":".join(
        (
            str(record.get("difficulty", "unknown")),
            str(record.get("opponent", "unknown")),
            distance,
            altitude,
            corner,
            self_zone,
            f"ea{int(_number(enemy, 'action'))}",
            projectile_bucket,
            f"h{self_hp}-{enemy_hp}",
        )
    )


def temporal_episode_split(
    records: list[dict[str, object]], *, validation_fraction: float = 0.20
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    if len(records) < 2:
        raise ValueError("at least two transitions are required")
    episodes = list(
        dict.fromkeys(str(record.get("episode_id", "")) for record in records)
    )
    if len(episodes) >= 2:
        validation_episodes = max(1, round(len(episodes) * validation_fraction))
        held_out = set(episodes[-validation_episodes:])
        train = [
            index for index, record in enumerate(records)
            if str(record.get("episode_id", "")) not in held_out
        ]
        validation = [
            index for index, record in enumerate(records)
            if str(record.get("episode_id", "")) in held_out
        ]
        if train and validation:
            return train, validation
    split = max(1, min(len(records) - 1, round(len(records) * (1.0 - validation_fraction))))
    return list(range(split)), list(range(split, len(records)))


def select_rows(values: list[object], indices: Iterable[int]) -> list[object]:
    return [values[index] for index in indices]
