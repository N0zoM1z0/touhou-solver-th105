"""Tiny online response table for character/action-specific defense.

This is deliberately not a trained policy.  It records direct outcomes of a
small set of legal responses and deterministically reuses the safest observed
one.  Unknown actions retain a conservative response-order prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .reward import basis_points, damage_bin_index, upper_tail_mean_bp


@dataclass
class ResponseStats:
    trials: int = 0
    successes: int = 0
    damage_events: int = 0
    total_damage: int = 0
    guard_events: int = 0
    total_guard_cost: int = 0
    normalized_samples: int = 0
    total_damage_bp: int = 0
    total_guard_cost_bp: int = 0
    damage_histogram: list[int] = field(default_factory=lambda: [0] * 8)

    @property
    def mean_damage(self) -> float:
        return self.total_damage / max(1, self.trials)

    @property
    def loss(self) -> float:
        trials = max(1, self.trials)
        if self.normalized_samples:
            mean_damage = self.total_damage_bp / self.normalized_samples
            mean_guard = self.total_guard_cost_bp / self.normalized_samples
        else:
            # Migration for old TH105 checkpoints (10000 HP / 1000 spirit).
            mean_damage = self.total_damage / trials
            mean_guard = self.total_guard_cost * 10.0 / trials
        tail = upper_tail_mean_bp(self.damage_histogram)
        excess_tail = max(0.0, tail - mean_damage)
        failure_rate = self.damage_events / trials
        success_rate = self.successes / trials
        return (
            1.60 * mean_damage
            + 0.45 * excess_tail
            + 450.0 * failure_rate
            + 0.12 * mean_guard
            - 50.0 * success_rate
        )


def normalize_legacy_response_stats(stats: ResponseStats) -> None:
    """Backfill v1 basis-point evidence for imported TH105 raw trials."""
    legacy_samples = max(0, stats.trials - stats.normalized_samples)
    if legacy_samples <= 0:
        return
    residual_damage = max(0, stats.total_damage - stats.total_damage_bp)
    residual_guard_bp = max(
        0, stats.total_guard_cost * 10 - stats.total_guard_cost_bp
    )
    observed_damage_events = sum(stats.damage_histogram[1:])
    legacy_damage_events = min(
        legacy_samples, max(0, stats.damage_events - observed_damage_events)
    )
    stats.damage_histogram[0] += legacy_samples - legacy_damage_events
    if legacy_damage_events:
        mean_damage_bp = round(residual_damage / legacy_damage_events)
        stats.damage_histogram[damage_bin_index(mean_damage_bp)] += (
            legacy_damage_events
        )
    stats.total_damage_bp += residual_damage
    stats.total_guard_cost_bp += residual_guard_bp
    stats.normalized_samples += legacy_samples


@dataclass
class DefenseEpisode:
    signature: str
    response: str
    start_hp: int
    start_spirit: int
    last_hp: int
    last_spirit: int
    minimum_distance: float
    damage: int = 0
    guard_cost: int = 0
    applied: bool = False
    max_hp: int = 10000
    max_spirit: int = 1000


class DefenseResponseModel:
    """Outcome memory keyed by an opaque action signature."""

    def __init__(self) -> None:
        self.table: dict[str, dict[str, ResponseStats]] = {}
        self.episode: DefenseEpisode | None = None
        self.choice_cache: dict[tuple[str, tuple[str, ...]], str] = {}

    def choose(self, signature: str, responses: tuple[str, ...]) -> str:
        if not responses:
            raise ValueError("at least one defense response is required")
        cache_key = (signature, responses)
        if cache_key in self.choice_cache:
            return self.choice_cache[cache_key]
        row = self.table.setdefault(signature, {})
        stats = {name: row.setdefault(name, ResponseStats()) for name in responses}

        # A response that has survived contact without ever taking damage is
        # preferred over exploration. This is what makes behavior converge.
        proven = [
            (item.loss, index, name)
            for index, name in enumerate(responses)
            if (item := stats[name]).successes > 0 and item.damage_events == 0
        ]
        if proven:
            choice = min(proven)[2]
            self.choice_cache[cache_key] = choice
            return choice

        # Keep the caller's conservative ordering for a brand-new action. Once
        # it has failed, try each still-unobserved alternative exactly once.
        if all(item.trials == 0 for item in stats.values()):
            self.choice_cache[cache_key] = responses[0]
            return responses[0]
        if any(item.damage_events for item in stats.values()):
            for name in responses:
                if stats[name].trials == 0:
                    self.choice_cache[cache_key] = name
                    return name
        choice = min(
            (item.loss, index, name)
            for index, name in enumerate(responses)
            for item in (stats[name],)
        )[2]
        self.choice_cache[cache_key] = choice
        return choice

    def begin(
        self,
        signature: str,
        response: str,
        *,
        hp: int,
        spirit: int,
        distance: float,
        max_hp: int = 10000,
        max_spirit: int = 1000,
    ) -> None:
        if self.episode is not None:
            self.finish()
        self.episode = DefenseEpisode(
            signature,
            response,
            hp,
            spirit,
            hp,
            spirit,
            distance,
            max_hp=max(1, max_hp),
            max_spirit=max(1, max_spirit),
        )

    def observe(self, *, hp: int, spirit: int, distance: float) -> None:
        episode = self.episode
        if episode is None:
            return
        if hp < episode.last_hp:
            episode.damage += episode.last_hp - hp
        if spirit < episode.last_spirit and hp >= episode.last_hp:
            episode.guard_cost += episode.last_spirit - spirit
        episode.last_hp = hp
        episode.last_spirit = spirit
        episode.minimum_distance = min(episode.minimum_distance, distance)

    def mark_applied(self) -> None:
        if self.episode is not None:
            self.episode.applied = True

    def finish(self, *, contact_distance: float = 150.0) -> None:
        episode = self.episode
        self.episode = None
        if episode is None or not episode.applied:
            return
        made_contact = episode.minimum_distance <= contact_distance or episode.guard_cost > 0
        if not made_contact and episode.damage == 0:
            return
        stats = self.table.setdefault(episode.signature, {}).setdefault(
            episode.response, ResponseStats()
        )
        stats.trials += 1
        damage_bp = basis_points(episode.damage, episode.max_hp)
        stats.normalized_samples += 1
        stats.total_damage_bp += damage_bp
        stats.total_guard_cost_bp += basis_points(
            episode.guard_cost, episode.max_spirit
        )
        stats.damage_histogram[damage_bin_index(damage_bp)] += 1
        if episode.damage:
            stats.damage_events += 1
            stats.total_damage += episode.damage
        else:
            stats.successes += 1
        if episode.guard_cost:
            stats.guard_events += 1
            stats.total_guard_cost += episode.guard_cost
        self.choice_cache = {
            key: value
            for key, value in self.choice_cache.items()
            if key[0] != episode.signature
        }

    def discard_episode(self) -> None:
        self.episode = None

    def export_state(self) -> dict[str, object]:
        return {
            signature: {response: asdict(stats) for response, stats in row.items()}
            for signature, row in self.table.items()
        }

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        self.choice_cache.clear()
        for signature, raw_row in state.items():
            if not isinstance(raw_row, dict):
                continue
            row = self.table.setdefault(str(signature), {})
            for response, raw in raw_row.items():
                if not isinstance(raw, dict):
                    continue
                histogram = raw.get("damage_histogram", [0] * 8)
                normalized_histogram = (
                    [max(0, int(value)) for value in histogram[:8]]
                    if isinstance(histogram, list) else [0] * 8
                )
                normalized_histogram += [0] * (8 - len(normalized_histogram))
                stats = ResponseStats(
                    trials=int(raw.get("trials", 0)),
                    successes=int(raw.get("successes", 0)),
                    damage_events=int(raw.get("damage_events", 0)),
                    total_damage=int(raw.get("total_damage", 0)),
                    guard_events=int(raw.get("guard_events", 0)),
                    total_guard_cost=int(raw.get("total_guard_cost", 0)),
                    normalized_samples=int(raw.get("normalized_samples", 0)),
                    total_damage_bp=int(raw.get("total_damage_bp", 0)),
                    total_guard_cost_bp=int(raw.get("total_guard_cost_bp", 0)),
                    damage_histogram=normalized_histogram,
                )
                normalize_legacy_response_stats(stats)
                row[str(response)] = stats

    def metrics(self) -> dict[str, object]:
        return {
            "actions": self.export_state(),
            "active": asdict(self.episode) if self.episode else None,
        }
