"""Compact one-row-per-round evidence for autonomous learning curves."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .reward import basis_points
from .schema import (
    ACTION_SCHEMA_VERSION,
    LEARNING_CURVE_SCHEMA_VERSION,
    TRAINING_GENERATION,
)


def load_generation_manifest(runtime_dir: Path) -> dict[str, object]:
    path = runtime_dir / "th105_generation.json"
    if not path.is_file():
        return {
            "training_generation": TRAINING_GENERATION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training generation manifest must be an object")
    if (
        str(raw.get("training_generation", "")) != TRAINING_GENERATION
        or int(raw.get("action_schema_version", 0)) != ACTION_SCHEMA_VERSION
    ):
        raise ValueError("runtime belongs to an incompatible training generation")
    return raw


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _integer(row: object, name: str) -> int:
    return int(row.get(name, 0)) if isinstance(row, dict) else 0


def _sum_prefix(counts: dict[str, object], prefix: str) -> int:
    return sum(int(value) for name, value in counts.items() if name.startswith(prefix))


def learning_curve_point(
    *,
    session_id: str,
    terminal_sequence: int,
    opponent: str,
    difficulty: str,
    playstyle: str,
    won: bool | None,
    me_hp: int,
    enemy_hp: int,
    max_me_hp: int,
    max_enemy_hp: int,
    policy_sha256: str | None,
    offline_policy_sha256: str | None,
    plugin_metrics: object,
) -> dict[str, object]:
    """Build a bounded, reward-independent physical learning checkpoint."""
    metrics = _mapping(plugin_metrics)
    counts = _mapping(metrics.get("counts"))
    coverage = _mapping(metrics.get("coverage_explorer"))
    cancel = _mapping(metrics.get("cancel_graph"))
    offense_state = _mapping(metrics.get("offense_outcome_state"))
    global_offense = _mapping(offense_state.get("*"))
    offense_rows = [
        row
        for label, row in global_offense.items()
        if label != "__reward__" and isinstance(row, dict)
    ]
    trials = sum(_integer(row, "trials") for row in offense_rows)
    connections = sum(_integer(row, "connections") for row in offense_rows)
    punished = sum(_integer(row, "punished_trials") for row in offense_rows)
    damage = sum(_integer(row, "total_damage_bp") for row in offense_rows)
    self_damage = sum(
        _integer(row, "total_self_damage_bp") for row in offense_rows
    )
    rounds = _mapping(_mapping(offense_state.get("__reward__")).get("rounds"))
    return {
        "learning_curve_schema_version": LEARNING_CURVE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "training_generation": TRAINING_GENERATION,
        "time": time.time(),
        "session_id": session_id,
        "terminal_sequence": int(terminal_sequence),
        "opponent": opponent,
        "difficulty": difficulty,
        "playstyle": playstyle,
        "policy_sha256": policy_sha256,
        "offline_policy_sha256": offline_policy_sha256,
        "outcome": {
            "won": won,
            "me_hp_bp": basis_points(me_hp, max(1, max_me_hp)),
            "enemy_hp_bp": basis_points(enemy_hp, max(1, max_enemy_hp)),
            "hp_differential_bp": basis_points(me_hp, max(1, max_me_hp))
            - basis_points(enemy_hp, max(1, max_enemy_hp)),
        },
        "cumulative": {
            "rounds": _integer(rounds, "rounds"),
            "wins": _integer(rounds, "wins"),
            "losses": _integer(rounds, "losses"),
            "draws": _integer(rounds, "draws"),
            "offense_actions": len(offense_rows),
            "offense_trials": trials,
            "connections": connections,
            "connection_rate": connections / trials if trials else 0.0,
            "punished_rate": punished / trials if trials else 0.0,
            "mean_damage_bp": damage / trials if trials else 0.0,
            "mean_self_damage_bp": self_damage / trials if trials else 0.0,
            "grammar_probe_attempts": _sum_prefix(
                counts, "grammar_probe_attempts:"
            ),
            "motion_probe_attempts": _sum_prefix(
                counts, "motion_probe_attempts:"
            ),
            "cancel_hypothesis_attempts": _sum_prefix(
                counts, "cancel_hypothesis_attempts:"
            ),
            "learned_cancel_attempts": _sum_prefix(
                counts, "learned_cancel_attempts:"
            ),
            "cancel_edges": int(cancel.get("edges", 0)),
            "accepted_cancel_transitions": int(
                cancel.get("accepted_transitions", 0)
            ),
            "coverage": float(coverage.get("coverage", 0.0)),
            "covered_scope_actions": int(coverage.get("covered_scope_actions", 0)),
            "available_scope_actions": int(
                coverage.get("available_scope_actions", 0)
            ),
        },
    }

