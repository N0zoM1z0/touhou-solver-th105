"""Runtime-safe reader and scalarizer for distilled offline outcome tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reward import DEFAULT_REWARD, RewardConfig, basis_points


DISTILLED_ARTIFACT_SCHEMA_VERSION = 1


def _bucket_context(
    *,
    difficulty: str,
    opponent: str,
    relative_x_q4: float,
    relative_y_q4: float,
    enemy_x_q4: float,
    enemy_action: int,
    enemy_projectile_count: int,
    self_hp_bp: int,
    enemy_hp_bp: int,
) -> str:
    distance = (
        "close" if abs(relative_x_q4) < 20
        else "mid" if abs(relative_x_q4) < 65
        else "far"
    )
    altitude = "ground" if abs(relative_y_q4) < 11 else "air"
    corner = "corner" if enemy_x_q4 < 38 or enemy_x_q4 > 282 else "field"
    projectile_bucket = (
        "p0" if enemy_projectile_count == 0
        else "p1-3" if enemy_projectile_count <= 3
        else "p4-8" if enemy_projectile_count <= 8
        else "p9+"
    )
    return ":".join(
        (
            difficulty,
            opponent,
            distance,
            altitude,
            corner,
            f"ea{enemy_action}",
            projectile_bucket,
            f"h{self_hp_bp // 1000}-{enemy_hp_bp // 1000}",
        )
    )


def live_distillation_context(
    *,
    difficulty: str,
    opponent: str,
    state: Any,
    enemy_projectiles: tuple[Any, ...] = (),
) -> str:
    me, enemy = state.p1, state.p2
    return _bucket_context(
        difficulty=difficulty,
        opponent=opponent,
        relative_x_q4=round((float(enemy.x) - float(me.x)) / 4.0),
        relative_y_q4=round((float(enemy.y) - float(me.y)) / 4.0),
        enemy_x_q4=round(float(enemy.x) / 4.0),
        enemy_action=int(enemy.action_id),
        enemy_projectile_count=len(enemy_projectiles),
        self_hp_bp=basis_points(int(me.hp), max(1, int(me.max_hp))),
        enemy_hp_bp=basis_points(int(enemy.hp), max(1, int(enemy.max_hp))),
    )


def predicted_outcome_utility(
    outcomes: dict[str, object], config: RewardConfig = DEFAULT_REWARD
) -> float:
    def number(name: str) -> float:
        try:
            return float(outcomes.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    damage = max(0.0, number("damage_bp"))
    self_damage = max(0.0, number("self_damage_bp"))
    tail = max(self_damage, number("self_damage_p90_bp"))
    spirit = max(0.0, number("spirit_cost_bp"))
    connection = max(0.0, min(1.0, number("connection_probability")))
    punished = max(0.0, min(1.0, number("punished_probability")))
    commitment_seconds = max(0.0, number("commitment_frames")) / 60.0
    terminal = max(-1.0, min(1.0, number("terminal_value")))
    return (
        config.damage_weight * damage
        - config.self_damage_weight * self_damage
        - config.spirit_weight * spirit
        - config.whiff_cost_bp * (1.0 - connection)
        - config.punished_cost_bp * punished
        - config.commitment_cost_bp_per_second * commitment_seconds
        - config.tail_risk_weight * max(0.0, tail - self_damage)
        + config.terminal_weight_bp * terminal
    )


@dataclass(frozen=True)
class DistilledScore:
    context: str
    action: str
    support: int
    outcomes: dict[str, float]
    utility: float
    adjustment: float


class DistilledOutcomePolicy:
    """Bounded lookup with zero-adjustment fallback for unknown contexts."""

    def __init__(self, data: object = None) -> None:
        raw = data if isinstance(data, dict) else {}
        if raw and int(raw.get("artifact_schema_version", 0)) != DISTILLED_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported distilled artifact schema")
        contexts = raw.get("contexts", {})
        self.contexts = contexts if isinstance(contexts, dict) else {}
        compatibility = raw.get("compatibility", {})
        self.compatibility = compatibility if isinstance(compatibility, dict) else {}
        self.hits = 0
        self.misses = 0
        self.last: DistilledScore | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        game_build_sha256: str | None = None,
        difficulty: str | None = None,
    ) -> "DistilledOutcomePolicy":
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = cls(data)
        builds = policy.compatibility.get("game_build_sha256", [])
        if game_build_sha256 and isinstance(builds, list) and builds:
            if game_build_sha256 not in {str(value) for value in builds}:
                raise ValueError("offline artifact game build mismatch")
        difficulties = policy.compatibility.get("difficulties", [])
        if difficulty and isinstance(difficulties, list) and difficulties:
            if difficulty not in {str(value) for value in difficulties}:
                raise ValueError("offline artifact difficulty mismatch")
        return policy

    def score(self, context: str, action: str) -> DistilledScore | None:
        row = self.contexts.get(context)
        candidate = row.get(action) if isinstance(row, dict) else None
        if not isinstance(candidate, dict):
            self.misses += 1
            self.last = None
            return None
        raw_outcomes = candidate.get("outcomes", {})
        if not isinstance(raw_outcomes, dict):
            self.misses += 1
            self.last = None
            return None
        outcomes: dict[str, float] = {}
        for name, value in raw_outcomes.items():
            try:
                outcomes[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        support = max(0, int(candidate.get("support", 0)))
        utility = predicted_outcome_utility(outcomes)
        confidence = min(1.0, support / 20.0)
        adjustment = max(-1200.0, min(1200.0, utility)) * confidence
        result = DistilledScore(
            context=context,
            action=action,
            support=support,
            outcomes=outcomes,
            utility=utility,
            adjustment=adjustment,
        )
        self.hits += 1
        self.last = result
        return result

    def metrics(self) -> dict[str, object]:
        return {
            "loaded": bool(self.contexts),
            "contexts": len(self.contexts),
            "hits": self.hits,
            "misses": self.misses,
            "last": (
                {
                    "context": self.last.context,
                    "action": self.last.action,
                    "support": self.last.support,
                    "utility": self.last.utility,
                    "adjustment": self.last.adjustment,
                }
                if self.last else None
            ),
        }
