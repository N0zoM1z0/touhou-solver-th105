"""Stable, deliberately small ABI between the battle shell and policy plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


POLICY_API_VERSION = 1


@dataclass(frozen=True)
class PolicyObservation:
    frame: int
    state: Any
    previous_state: Any | None
    enemy_projectiles: tuple[Any, ...]
    # Added compatibly at the tail: old policies ignore it, while a freshly
    # started battle shell can expose Sakuya's live knife layout.
    own_projectiles: tuple[Any, ...] = ()
    prior_opponent_model: dict[str, object] = field(default_factory=dict)
    prior_projectile_model: dict[str, object] = field(default_factory=dict)
    prior_defense_model: dict[str, object] = field(default_factory=dict)
    prior_offense_model: dict[str, object] = field(default_factory=dict)
    prior_human_demonstrations: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    keys: frozenset[str]
    intent: str


class CombatPolicy(Protocol):
    api_version: int
    name: str

    def decide(self, observation: PolicyObservation) -> PolicyDecision: ...

    def metrics(self) -> dict[str, int | float | str | None]: ...
