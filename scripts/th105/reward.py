"""Versioned, dimensionless reward primitives for compact online learning.

The learner stores native outcome components rather than only a scalar value.
That keeps reward changes reversible and leaves the same corpus usable by a
future offline learner without putting heavyweight inference in the game loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


REWARD_VERSION = 1
SELF_DAMAGE_BIN_UPPER_BP = (0, 100, 250, 500, 1000, 2000, 4000, 10000)


def basis_points(value: int | float, maximum: int | float) -> int:
    """Return a bounded fraction in 1/10000 units."""
    if maximum <= 0:
        return 0
    return max(0, min(10000, round(float(value) * 10000.0 / float(maximum))))


def damage_bin_index(damage_bp: int) -> int:
    for index, upper in enumerate(SELF_DAMAGE_BIN_UPPER_BP):
        if damage_bp <= upper:
            return index
    return len(SELF_DAMAGE_BIN_UPPER_BP) - 1


def upper_tail_mean_bp(histogram: Sequence[int], *, tail_fraction: float = 0.20) -> float:
    """Conservative CVaR-like estimate from eight compact histogram counters."""
    total = sum(max(0, int(count)) for count in histogram)
    if total <= 0:
        return 0.0
    remaining = max(1.0, total * max(0.01, min(1.0, tail_fraction)))
    weighted = 0.0
    consumed = 0.0
    for upper, raw_count in reversed(tuple(zip(SELF_DAMAGE_BIN_UPPER_BP, histogram))):
        take = min(remaining - consumed, max(0, int(raw_count)))
        if take <= 0:
            continue
        weighted += float(upper) * take
        consumed += take
        if consumed >= remaining:
            break
    return weighted / max(1.0, consumed)


@dataclass(frozen=True)
class RewardConfig:
    """Small v1 scalarization; all inputs are dimensionless basis points."""

    damage_weight: float = 1.00
    self_damage_weight: float = 1.60
    spirit_weight: float = 0.12
    whiff_cost_bp: float = 350.0
    punished_cost_bp: float = 450.0
    commitment_cost_bp_per_second: float = 60.0
    tail_risk_weight: float = 0.45
    terminal_weight_bp: float = 1400.0


DEFAULT_REWARD = RewardConfig()


def empirical_action_value(stats: object, config: RewardConfig = DEFAULT_REWARD) -> float:
    """Score aggregate sufficient statistics without needing raw frame logs."""
    trials = max(1, int(getattr(stats, "trials", 0)))
    normalized_samples = int(getattr(stats, "normalized_samples", 0))
    if normalized_samples > 0:
        damage_bp = float(getattr(stats, "total_damage_bp", 0)) / normalized_samples
        self_damage_bp = (
            float(getattr(stats, "total_self_damage_bp", 0)) / normalized_samples
        )
        spirit_bp = (
            float(getattr(stats, "total_spirit_cost_bp", 0)) / normalized_samples
        )
    else:
        # TH105 normally uses 10000 HP and 1000 spirit. This migration path
        # keeps old checkpoints useful until each row receives a v1 sample.
        damage_bp = float(getattr(stats, "total_damage", 0)) / trials
        self_damage_bp = float(getattr(stats, "total_self_damage", 0)) / trials
        spirit_bp = float(getattr(stats, "total_spirit_cost", 0)) * 10.0 / trials

    connections = int(getattr(stats, "connections", 0))
    punished = int(getattr(stats, "punished_trials", 0))
    commitment_seconds = (
        float(getattr(stats, "total_commitment", 0)) / trials / 60.0
    )
    histogram = getattr(stats, "self_damage_histogram", ())
    tail_bp = (
        upper_tail_mean_bp(histogram)
        if isinstance(histogram, (list, tuple)) and len(histogram) == 8
        else self_damage_bp
    )
    excess_tail = max(0.0, tail_bp - self_damage_bp)
    terminal_credit = (
        float(getattr(stats, "terminal_win_credit", 0.0))
        - float(getattr(stats, "terminal_loss_credit", 0.0))
    ) / trials
    return (
        config.damage_weight * damage_bp
        - config.self_damage_weight * self_damage_bp
        - config.spirit_weight * spirit_bp
        - config.whiff_cost_bp * max(0, trials - connections) / trials
        - config.punished_cost_bp * punished / trials
        - config.commitment_cost_bp_per_second * commitment_seconds
        - config.tail_risk_weight * excess_tail
        + config.terminal_weight_bp * terminal_credit
    )
