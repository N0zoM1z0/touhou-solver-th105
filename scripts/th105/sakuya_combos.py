"""Guide-derived Sakuya combo catalogue and context scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .motion import build_beginner_ground_combo, build_motion_frames


@dataclass(frozen=True)
class ComboContext:
    distance: float
    enemy_x: float
    enemy_y: float
    spirit: int
    punish: bool = False

    @property
    def cornered(self) -> bool:
        return self.enemy_x < 150.0 or self.enemy_x > 1130.0


@dataclass(frozen=True)
class ComboCandidate:
    name: str
    estimated_damage: int
    spirit_cost: int
    max_distance: float
    corner_only: bool
    builder: Callable[[str], list[set[str]]]


def _pulse(chord: set[str], pressed: int = 2, released: int = 6) -> list[set[str]]:
    return [set(chord) for _ in range(pressed)] + [set() for _ in range(released)]


def _aaaa(_toward: str) -> list[set[str]]:
    frames: list[set[str]] = []
    for release in (6, 6, 8, 10):
        frames.extend(_pulse({"z"}, released=release))
    return frames


def _aaaa_623b(toward: str) -> list[set[str]]:
    return _aaaa(toward) + build_motion_frames("623", "x", toward, recovery_frames=8)


def _two_a_route(toward: str) -> list[set[str]]:
    frames = _pulse({"down", "z"}, released=8)
    frames += _pulse({"down", "x"}, pressed=3, released=9)
    frames += _pulse({"c"}, pressed=3, released=9)
    frames += build_motion_frames("236", "x", toward, recovery_frames=8)
    return frames


COMBOS: tuple[ComboCandidate, ...] = (
    ComboCandidate(
        "AAA-2B-C-236B",
        estimated_damage=2297,
        spirit_cost=600,
        # Runtime sampling: close-A action 300 appeared at 42--60 units,
        # while 91--96 units selected far-A action 301 (which cannot start
        # this route).  Keep a small interpolation margin only.
        max_distance=64.0,
        corner_only=False,
        builder=build_beginner_ground_combo,
    ),
    ComboCandidate(
        "2A-2B-C-236B",
        estimated_damage=1956,
        spirit_cost=600,
        max_distance=92.0,
        corner_only=False,
        builder=_two_a_route,
    ),
    ComboCandidate(
        "AAAA-623B",
        estimated_damage=2101,
        spirit_cost=200,
        max_distance=64.0,
        corner_only=True,
        builder=_aaaa_623b,
    ),
    ComboCandidate(
        "AAAA-meterless",
        estimated_damage=1500,
        spirit_cost=0,
        max_distance=64.0,
        corner_only=False,
        builder=_aaaa,
    ),
)


def select_combo(
    context: ComboContext,
    *,
    reserve_spirit: int = 200,
    learned_adjustments: dict[str, float] | None = None,
) -> ComboCandidate | None:
    """Choose damage subject to range, geometry and post-combo reserve."""
    viable = rank_combos(
        context,
        reserve_spirit=reserve_spirit,
        learned_adjustments=learned_adjustments,
    )
    return viable[0][1] if viable else None


def rank_combos(
    context: ComboContext,
    *,
    reserve_spirit: int = 200,
    learned_adjustments: dict[str, float] | None = None,
) -> tuple[tuple[float, ComboCandidate], ...]:
    """Return every native/context-gated route, highest base score first."""
    if context.enemy_y > 44.0:
        return ()
    viable: list[tuple[float, ComboCandidate]] = []
    for combo in COMBOS:
        if context.distance > combo.max_distance:
            continue
        if combo.corner_only and not context.cornered:
            continue
        if context.spirit - combo.spirit_cost < reserve_spirit:
            continue
        score = float(combo.estimated_damage)
        score -= combo.spirit_cost * 0.45
        if combo.corner_only and context.cornered:
            score += 220.0
        if context.punish:
            score += 120.0
        # Prefer the low-resource route when damage is close so defensive
        # spirit remains available for flight/guard recovery.
        score += min(300.0, context.spirit - combo.spirit_cost) * 0.15
        if learned_adjustments is not None:
            score += learned_adjustments.get(combo.name, 0.0)
        viable.append((score, combo))
    return tuple(sorted(viable, key=lambda item: (item[0], item[1].name), reverse=True))
