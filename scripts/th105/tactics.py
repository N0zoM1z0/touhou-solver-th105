"""Small explainable action scorer for finite-horizon combat choices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCandidate:
    name: str
    startup: float
    commitment: float
    damage: float
    spirit_cost: int
    minimum_range: float
    maximum_range: float
    role: str = "strike"
    punish_rate: float = 0.0


@dataclass(frozen=True)
class TacticalContext:
    distance: float
    vertical_gap: float
    spirit: int
    enemy_phase: str
    punish_window: float
    own_projectiles: int
    enemy_closing: bool
    enemy_time_to_contact: float = float("inf")
    reserve_spirit: int = 200


@dataclass(frozen=True)
class ScoredAction:
    candidate: ActionCandidate
    score: float
    hit_probability: float
    reason: str


def score_action(candidate: ActionCandidate, context: TacticalContext) -> ScoredAction:
    if context.spirit - candidate.spirit_cost < context.reserve_spirit:
        return ScoredAction(candidate, float("-inf"), 0.0, "spirit-reserve")
    if not candidate.minimum_range <= context.distance <= candidate.maximum_range:
        return ScoredAction(candidate, float("-inf"), 0.0, "range")
    if context.enemy_phase == "spell-danger":
        return ScoredAction(candidate, float("-inf"), 0.0, "enemy-spell")

    if context.enemy_phase == "reaction":
        hit_probability = 0.90
    elif context.enemy_phase == "recovery":
        if candidate.startup + 2.0 > context.punish_window:
            return ScoredAction(candidate, float("-inf"), 0.0, "late-punish")
        hit_probability = 0.94
    elif context.enemy_phase == "neutral":
        hit_probability = 0.48
    elif context.enemy_phase == "startup":
        hit_probability = 0.18
    else:
        hit_probability = 0.10

    if candidate.role == "anti-air":
        hit_probability += 0.22 if context.vertical_gap > 55.0 else -0.22
    elif candidate.role == "spread":
        hit_probability += 0.16 if context.vertical_gap > 45.0 else 0.03
    elif candidate.role == "setup":
        if context.own_projectiles < 3:
            hit_probability += 0.12
        else:
            hit_probability -= min(0.30, context.own_projectiles * 0.025)
        if context.enemy_closing:
            hit_probability += 0.08

    # Neutral does not mean harmless: a closing opponent can react during our
    # startup.  Reject setup that loses before its first knife exists, and
    # penalize other casts in proportion to the uncovered startup interval.
    intercept_penalty = 0.0
    if context.enemy_closing and context.enemy_phase == "neutral":
        slack = context.enemy_time_to_contact - candidate.startup
        if slack <= 2.0 and candidate.role == "setup":
            return ScoredAction(candidate, float("-inf"), 0.0, "startup-intercept")
        intercept_penalty = max(0.0, 8.0 - slack) * 45.0

    hit_probability = min(0.98, max(0.02, hit_probability))
    exposure_penalty = candidate.commitment * (
        4.0 if context.enemy_phase in {"startup", "active", "unknown"} else 1.5
    )
    resource_penalty = candidate.spirit_cost * 0.32
    learned_punish_penalty = min(1.0, max(0.0, candidate.punish_rate)) * 1600.0
    layout_value = 180.0 if candidate.role == "setup" and context.own_projectiles < 3 else 0.0
    score = (
        candidate.damage * hit_probability
        + layout_value
        - exposure_penalty
        - resource_penalty
        - intercept_penalty
        - learned_punish_penalty
    )
    return ScoredAction(candidate, score, hit_probability, "ok")


def rank_actions(
    candidates: tuple[ActionCandidate, ...], context: TacticalContext
) -> tuple[ScoredAction, ...]:
    return tuple(
        sorted(
            (score_action(candidate, context) for candidate in candidates),
            key=lambda item: item.score,
            reverse=True,
        )
    )
