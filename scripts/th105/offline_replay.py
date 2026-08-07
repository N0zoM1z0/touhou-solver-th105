"""Rebuild v6 online outcome priors from reusable physical transitions."""

from __future__ import annotations

from dataclasses import dataclass

from .offense_learning import ActionOutcomeModel, OptionOutcomeModel, combat_context
from .opponent_model import OpponentActionModel


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _integer(mapping: dict[str, object], name: str, default: int = 0) -> int:
    try:
        return int(mapping.get(name, default))
    except (TypeError, ValueError):
        return default


def _action_family(
    action_id: int,
    enemy: dict[str, object],
    profile: dict[str, object],
) -> str:
    if 50 <= action_id < 200:
        return "reaction"
    if 600 <= action_id < 800:
        return "spell"
    if not OpponentActionModel.is_offensive(action_id):
        return "neutral"
    strike = (
        _integer(enemy, "attack_box_count") > 0
        or _integer(profile, "active_box_observations") > 0
    )
    projectile = _integer(profile, "projectile_samples") > 0
    if strike and projectile:
        return "hybrid"
    if projectile:
        return "projectile"
    if strike:
        return "strike"
    return "offensive"


def _action_phase(
    action_id: int,
    enemy: dict[str, object],
    profile: dict[str, object],
) -> str:
    """Conservative phase reconstruction for corpus rows predating v6."""
    if 50 <= action_id < 200:
        return "reaction"
    if not OpponentActionModel.is_offensive(action_id):
        return "neutral"
    if 600 <= action_id < 800:
        return "spell-danger"
    if _integer(enemy, "attack_box_count") > 0:
        return "active"
    action_frame = _integer(enemy, "action_frame")
    last_threat = max(
        float(profile.get("last_impact", 0.0) or 0.0),
        float(profile.get("last_projectile", 0.0) or 0.0),
        float(profile.get("last_active_box", 0.0) or 0.0),
    )
    # action_frame is pose-local in the supported build.  It can prove an
    # observed late frame, but cannot safely prove startup; unknown offense is
    # therefore kept in the shared danger bucket.
    if last_threat > 0.0 and action_frame > last_threat + 2.0:
        return "recovery"
    return "unknown"


def replay_context(record: dict[str, object], profiles: dict[str, object]) -> str:
    saved = record.get("learning_context")
    if isinstance(saved, str) and "|ef=" in saved:
        return saved
    state = _mapping(record.get("state"))
    me = _mapping(state.get("self"))
    enemy = _mapping(state.get("enemy"))
    action_id = _integer(enemy, "action")
    raw_profile = profiles.get(str(action_id), {})
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    return combat_context(
        distance=abs(_integer(state, "relative_x_q4")) * 4.0,
        enemy_y=_integer(enemy, "y_q4") * 4.0,
        enemy_x=_integer(enemy, "x_q4") * 4.0,
        player_y=_integer(me, "y_q4") * 4.0,
        phase=_action_phase(action_id, enemy, profile),
        enemy_action=(
            action_id if OpponentActionModel.is_offensive(action_id) else None
        ),
        enemy_action_family=_action_family(action_id, enemy, profile),
    )


def replay_action_label(record: dict[str, object]) -> str | None:
    """Map recorded policy intent back to the online outcome-model label."""
    action = str(record.get("action", ""))
    state = _mapping(record.get("state"))
    me = _mapping(state.get("self"))
    if action.startswith("motion-probe:"):
        return (
            f"motion:{me.get('character_vtable', 'unknown')}:{action.split(':', 1)[1]}"
        )
    if action.startswith("learned-cancel:"):
        return f"cancel:{action.split(':', 1)[1]}"
    if action.startswith("cancel-discovery:"):
        source = ":".join(
            (
                str(me.get("character_vtable", "unknown")),
                str(_integer(me, "action")),
                str(_integer(me, "sequence")),
                str(_integer(me, "pose")),
            )
        )
        return f"cancel-hypothesis:{source}:{action.split(':', 1)[1]}"
    if action.startswith(("grammar-probe:", "human-probe:")):
        return action
    return None


def _sample(record: dict[str, object]) -> dict[str, object]:
    state = _mapping(record.get("state"))
    next_state = _mapping(record.get("next_state"))
    me = _mapping(state.get("self"))
    next_me = _mapping(next_state.get("self"))
    outcome = _mapping(record.get("outcome"))
    own_projectiles = state.get("own_projectiles")
    next_projectiles = next_state.get("own_projectiles")
    start_count = len(own_projectiles) if isinstance(own_projectiles, list) else 0
    end_count = len(next_projectiles) if isinstance(next_projectiles, list) else 0
    terminal = outcome.get("terminal")
    return {
        "damage": _integer(outcome, "damage"),
        "self_damage": _integer(outcome, "self_damage"),
        "spirit_cost": max(0, -_integer(outcome, "spirit_delta")),
        "commitment": max(1, _integer(record, "duration_frames", 1)),
        "damage_bp": _integer(outcome, "damage_bp"),
        "self_damage_bp": _integer(outcome, "self_damage_bp"),
        "spirit_cost_bp": _integer(outcome, "spirit_cost_bp"),
        "displacement": abs((_integer(next_me, "x_q4") - _integer(me, "x_q4")) * 4.0),
        "action_changed": bool(
            _integer(next_me, "action") >= 300
            and _integer(next_me, "action") != _integer(me, "action")
        ),
        "projectile_spawned": end_count > start_count,
        "terminal": str(terminal) if terminal in {"win", "loss"} else None,
    }


@dataclass
class ReplayResult:
    transitions_seen: int
    transitions_used: int
    offense_samples: int
    option_samples: int
    opponents: dict[str, dict[str, object]]


def replay_transitions(
    records: list[dict[str, object]],
    knowledge: dict[str, object],
    *,
    self_character: str,
) -> ReplayResult:
    """Build fresh v6 sufficient statistics without changing the raw corpus."""
    characters = _mapping(knowledge.get("characters"))
    offense_models: dict[str, ActionOutcomeModel] = {}
    option_models: dict[str, OptionOutcomeModel] = {}
    used = offense_samples = option_samples = 0
    seen_ids: set[str] = set()
    filtered: list[dict[str, object]] = []
    for record in records:
        transition_id = str(record.get("transition_id", ""))
        if transition_id and transition_id in seen_ids:
            continue
        seen_ids.add(transition_id)
        state = _mapping(record.get("state"))
        me = _mapping(state.get("self"))
        if str(me.get("character_vtable", "")).upper() != self_character.upper():
            continue
        filtered.append(record)
    filtered.sort(
        key=lambda row: (
            str(row.get("session_id", "")),
            str(row.get("episode_id", "")),
            _integer(row, "step"),
        )
    )
    for record in filtered:
        opponent = str(record.get("opponent", "unknown"))
        entry = _mapping(characters.get(opponent))
        profiles = _mapping(entry.get("profiles"))
        context = replay_context(record, profiles)
        sample = _sample(record)
        label = replay_action_label(record)
        if label is not None:
            offense_models.setdefault(opponent, ActionOutcomeModel()).record_sample(
                label, context, **sample
            )
            offense_samples += 1
        if label is not None or isinstance(record.get("combat_option"), str):
            used += 1

    # Micro-intent boundaries are more frequent than the online option bandit.
    # Fold contiguous rows into bounded semi-Markov option segments so defense
    # frames do not outvote one deliberate attack merely by lasting longer.
    active_key: tuple[str, str, str] | None = None
    active_context = ""
    active_sample: dict[str, object] | None = None

    def flush_option() -> None:
        nonlocal active_key, active_context, active_sample, option_samples
        if active_key is None or active_sample is None:
            return
        opponent, _episode, option = active_key
        option_models.setdefault(opponent, OptionOutcomeModel()).record_sample(
            f"option:{option}", active_context, **active_sample
        )
        option_samples += 1
        active_key = None
        active_sample = None

    legal_options = {"defend", "reposition", "approach", "attack", "punish"}
    for record in filtered:
        option = record.get("combat_option")
        if not isinstance(option, str) or option not in legal_options:
            flush_option()
            continue
        opponent = str(record.get("opponent", "unknown"))
        episode = str(record.get("episode_id", ""))
        key = (opponent, episode, option)
        entry = _mapping(characters.get(opponent))
        profiles = _mapping(entry.get("profiles"))
        sample = _sample(record)
        duration = int(sample["commitment"])
        if (
            active_key != key
            or active_sample is None
            or int(active_sample["commitment"]) + duration > 60
        ):
            flush_option()
            active_key = key
            active_context = replay_context(record, profiles)
            active_sample = sample
            continue
        for name in (
            "damage",
            "self_damage",
            "spirit_cost",
            "commitment",
            "damage_bp",
            "self_damage_bp",
            "spirit_cost_bp",
        ):
            active_sample[name] = int(active_sample[name]) + int(sample[name])
        active_sample["displacement"] = max(
            float(active_sample["displacement"]), float(sample["displacement"])
        )
        active_sample["action_changed"] = bool(
            active_sample["action_changed"] or sample["action_changed"]
        )
        active_sample["projectile_spawned"] = bool(
            active_sample["projectile_spawned"] or sample["projectile_spawned"]
        )
        if sample["terminal"] is not None:
            active_sample["terminal"] = sample["terminal"]
    flush_option()
    opponents: dict[str, dict[str, object]] = {}
    for opponent in sorted(set(offense_models) | set(option_models)):
        opponents[opponent] = {
            "offense_outcomes": offense_models.get(
                opponent, ActionOutcomeModel()
            ).export_state(),
            "option_outcomes": option_models.get(
                opponent, OptionOutcomeModel()
            ).export_state(),
        }
    return ReplayResult(
        transitions_seen=len(records),
        transitions_used=used,
        offense_samples=offense_samples,
        option_samples=option_samples,
        opponents=opponents,
    )
