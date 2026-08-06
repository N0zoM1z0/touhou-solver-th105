"""Runtime-safe reader and scalarizer for distilled offline outcome tables."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reward import DEFAULT_REWARD, RewardConfig, basis_points


DISTILLED_ARTIFACT_SCHEMA_VERSION = 1
MAX_BACKOFF_DISTANCE = 12.0
MAX_BACKOFF_CACHE = 4096

_DISTANCE_ORDINAL = {"close": 0, "mid": 1, "far": 2}
_PROJECTILE_ORDINAL = {"p0": 0, "p1-3": 1, "p4-8": 2, "p9+": 3}


@dataclass(frozen=True)
class _ContextFeatures:
    difficulty: str
    opponent: str
    distance: int
    altitude: str
    corner: str
    enemy_action: int
    projectiles: int
    self_hp: int
    enemy_hp: int


def _parse_context(context: str) -> _ContextFeatures | None:
    parts = context.split(":")
    if len(parts) != 8:
        return None
    difficulty, opponent, distance, altitude, corner, action, projectiles, hp = parts
    if distance not in _DISTANCE_ORDINAL or projectiles not in _PROJECTILE_ORDINAL:
        return None
    if not action.startswith("ea") or not hp.startswith("h") or "-" not in hp:
        return None
    try:
        self_hp, enemy_hp = hp[1:].split("-", 1)
        return _ContextFeatures(
            difficulty=difficulty,
            opponent=opponent,
            distance=_DISTANCE_ORDINAL[distance],
            altitude=altitude,
            corner=corner,
            enemy_action=int(action[2:]),
            projectiles=_PROJECTILE_ORDINAL[projectiles],
            self_hp=int(self_hp),
            enemy_hp=int(enemy_hp),
        )
    except ValueError:
        return None


def _context_distance(left: _ContextFeatures, right: _ContextFeatures) -> float:
    return (
        2.0 * abs(left.distance - right.distance)
        + 3.0 * (left.altitude != right.altitude)
        + 1.5 * (left.corner != right.corner)
        + 4.0 * (left.enemy_action != right.enemy_action)
        + 1.5 * abs(left.projectiles - right.projectiles)
        + 0.35 * (abs(left.self_hp - right.self_hp) + abs(left.enemy_hp - right.enemy_hp))
    )


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
    matched_context: str
    action: str
    support: int
    outcomes: dict[str, float]
    utility: float
    adjustment: float
    backoff_distance: float


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
        self.exact_hits = 0
        self.backoff_hits = 0
        self.misses = 0
        self.last: DistilledScore | None = None
        self._rows_by_scope: dict[
            tuple[str, str, str],
            list[tuple[str, _ContextFeatures, dict[str, object]]],
        ] = {}
        self._backoff_cache: OrderedDict[
            tuple[str, str], tuple[str, dict[str, object], float] | None
        ] = OrderedDict()
        for context, actions in self.contexts.items():
            features = _parse_context(str(context))
            if features is None or not isinstance(actions, dict):
                continue
            for action, candidate in actions.items():
                if not isinstance(candidate, dict):
                    continue
                scope = (features.difficulty, features.opponent, str(action))
                self._rows_by_scope.setdefault(scope, []).append(
                    (str(context), features, candidate)
                )

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

    def _backoff(
        self, context: str, action: str
    ) -> tuple[str, dict[str, object], float] | None:
        cache_key = (context, action)
        if cache_key in self._backoff_cache:
            cached = self._backoff_cache.pop(cache_key)
            self._backoff_cache[cache_key] = cached
            return cached
        features = _parse_context(context)
        result: tuple[str, dict[str, object], float] | None = None
        if features is not None:
            scope = (features.difficulty, features.opponent, action)
            ranked: list[tuple[float, int, str, dict[str, object]]] = []
            for matched_context, candidate_features, candidate in self._rows_by_scope.get(
                scope, ()
            ):
                distance = _context_distance(features, candidate_features)
                if distance <= MAX_BACKOFF_DISTANCE:
                    ranked.append(
                        (
                            distance,
                            -max(0, int(candidate.get("support", 0))),
                            matched_context,
                            candidate,
                        )
                    )
            if ranked:
                distance, _support, matched_context, candidate = min(
                    ranked, key=lambda row: (row[0], row[1], row[2])
                )
                result = (matched_context, candidate, distance)
        if len(self._backoff_cache) >= MAX_BACKOFF_CACHE:
            self._backoff_cache.popitem(last=False)
        self._backoff_cache[cache_key] = result
        return result

    def score(self, context: str, action: str) -> DistilledScore | None:
        row = self.contexts.get(context)
        candidate = row.get(action) if isinstance(row, dict) else None
        matched_context = context
        backoff_distance = 0.0
        if not isinstance(candidate, dict):
            backed_off = self._backoff(context, action)
            if backed_off is not None:
                matched_context, candidate, backoff_distance = backed_off
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
            matched_context=matched_context,
            action=action,
            support=support,
            outcomes=outcomes,
            utility=utility,
            adjustment=adjustment,
            backoff_distance=backoff_distance,
        )
        self.hits += 1
        if backoff_distance == 0.0:
            self.exact_hits += 1
        else:
            self.backoff_hits += 1
        self.last = result
        return result

    def metrics(self) -> dict[str, object]:
        return {
            "loaded": bool(self.contexts),
            "contexts": len(self.contexts),
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "backoff_hits": self.backoff_hits,
            "misses": self.misses,
            "last": (
                {
                    "context": self.last.context,
                    "matched_context": self.last.matched_context,
                    "action": self.last.action,
                    "support": self.last.support,
                    "utility": self.last.utility,
                    "adjustment": self.last.adjustment,
                    "backoff_distance": self.last.backoff_distance,
                }
                if self.last else None
            ),
        }
