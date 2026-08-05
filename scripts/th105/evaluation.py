"""Reproducible descriptive evaluation of durable TH105 learning evidence."""

from __future__ import annotations

import math

from .defense_learning import DefenseResponseModel
from .offense_learning import ActionOutcomeModel
from .reward import upper_tail_mean_bp


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> list[float]:
    """Return a bounded binomial confidence interval without scipy."""
    if trials <= 0:
        return [0.0, 1.0]
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _integer(row: object, name: str) -> int:
    return int(row.get(name, 0)) if isinstance(row, dict) else 0


def _histogram(row: object, name: str) -> list[int]:
    raw = row.get(name, []) if isinstance(row, dict) else []
    values = [max(0, int(value)) for value in raw[:8]] if isinstance(raw, list) else []
    return values + [0] * (8 - len(values))


def _sum_histograms(rows: list[dict[str, object]], name: str) -> list[int]:
    output = [0] * 8
    for row in rows:
        for index, value in enumerate(_histogram(row, name)):
            output[index] += value
    return output


def evaluate_character(entry: object) -> dict[str, object]:
    raw = entry if isinstance(entry, dict) else {}
    # Evaluation must use the same lazy migration as the online agent. Reading
    # legacy rows directly mixes raw TH105 HP/spirit totals with basis points
    # and can make an otherwise healthy checkpoint look badly regressed.
    offense_model = ActionOutcomeModel()
    offense_model.import_state(raw.get("offense_outcomes", {}))
    offense_state = offense_model.export_state()
    global_offense = (
        offense_state.get("*", {}) if isinstance(offense_state, dict) else {}
    )
    offense_rows = [
        row for row in global_offense.values()
        if isinstance(row, dict)
    ] if isinstance(global_offense, dict) else []
    trials = sum(_integer(row, "trials") for row in offense_rows)
    connections = sum(_integer(row, "connections") for row in offense_rows)
    normalized = sum(_integer(row, "normalized_samples") for row in offense_rows)
    punished = sum(_integer(row, "punished_trials") for row in offense_rows)
    commitment = sum(_integer(row, "total_commitment") for row in offense_rows)
    offense_histogram = _sum_histograms(offense_rows, "self_damage_histogram")

    reward_row = (
        offense_state.get("__reward__", {})
        if isinstance(offense_state, dict) else {}
    )
    rounds = reward_row.get("rounds", {}) if isinstance(reward_row, dict) else {}
    round_count = _integer(rounds, "rounds")
    wins = _integer(rounds, "wins")

    defense_model = DefenseResponseModel()
    defense_model.import_state(raw.get("defense_responses", {}))
    defense_state = defense_model.export_state()
    defense_rows: list[dict[str, object]] = []
    if isinstance(defense_state, dict):
        for response_map in defense_state.values():
            if isinstance(response_map, dict):
                defense_rows.extend(
                    row for row in response_map.values() if isinstance(row, dict)
                )
    defense_trials = sum(_integer(row, "trials") for row in defense_rows)
    defense_successes = sum(_integer(row, "successes") for row in defense_rows)
    defense_damage_events = sum(_integer(row, "damage_events") for row in defense_rows)
    defense_normalized = sum(_integer(row, "normalized_samples") for row in defense_rows)
    defense_histogram = _sum_histograms(defense_rows, "damage_histogram")

    labels = list(global_offense) if isinstance(global_offense, dict) else []
    skill_labels = [
        label for label in labels
        if isinstance(label, str)
        and len(label) >= 4
        and label[:-1].isdigit()
        and label[-1:] in {"A", "B", "C"}
    ]
    return {
        "rounds": {
            "rounds": round_count,
            "wins": wins,
            "losses": _integer(rounds, "losses"),
            "draws": _integer(rounds, "draws"),
            "win_rate": wins / round_count if round_count else 0.0,
            "win_rate_ci95": wilson_interval(wins, round_count),
        },
        "offense": {
            "actions_observed": len(offense_rows),
            "skills_observed": len(skill_labels),
            "trials": trials,
            "connections": connections,
            "connection_rate": connections / trials if trials else 0.0,
            "connection_rate_ci95": wilson_interval(connections, trials),
            "punished_rate": punished / trials if trials else 0.0,
            "mean_damage_bp": (
                sum(_integer(row, "total_damage_bp") for row in offense_rows)
                / normalized if normalized else 0.0
            ),
            "mean_self_damage_bp": (
                sum(_integer(row, "total_self_damage_bp") for row in offense_rows)
                / normalized if normalized else 0.0
            ),
            "mean_commitment_frames": commitment / trials if trials else 0.0,
            "self_damage_upper_tail_bp": upper_tail_mean_bp(offense_histogram),
        },
        "defense": {
            "signatures": len(defense_state) if isinstance(defense_state, dict) else 0,
            "trials": defense_trials,
            "successes": defense_successes,
            "success_rate": (
                defense_successes / defense_trials if defense_trials else 0.0
            ),
            "success_rate_ci95": wilson_interval(defense_successes, defense_trials),
            "damage_event_rate": (
                defense_damage_events / defense_trials if defense_trials else 0.0
            ),
            "mean_damage_bp": (
                sum(_integer(row, "total_damage_bp") for row in defense_rows)
                / defense_normalized if defense_normalized else 0.0
            ),
            "damage_upper_tail_bp": upper_tail_mean_bp(defense_histogram),
        },
        "coverage": {
            "opponent_actions": len(raw.get("profiles", {}))
            if isinstance(raw.get("profiles"), dict) else 0,
            "attack_geometry_rows": len(raw.get("attack_geometry", {}))
            if isinstance(raw.get("attack_geometry"), dict) else 0,
            "cancel_edges": len(raw.get("cancel_graph", {}))
            if isinstance(raw.get("cancel_graph"), dict) else 0,
        },
    }


def evaluate_knowledge(data: object) -> dict[str, object]:
    root = data if isinstance(data, dict) else {}
    characters = root.get("characters", {})
    characters = characters if isinstance(characters, dict) else {}
    return {
        "evaluation_schema_version": 1,
        "source_schema_version": int(root.get("schema_version", 1)),
        "characters": {
            str(key): evaluate_character(entry)
            for key, entry in sorted(characters.items())
        },
        "limitations": [
            "descriptive evidence, not causal off-policy evaluation",
            "complete campaign A/B remains the deployment authority",
            "proxy regressions are diagnostic and do not automatically reject a policy",
        ],
    }
