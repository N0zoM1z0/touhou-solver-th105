"""Bounded sufficient statistics for attacks and combo outcomes."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from .reward import (
    REWARD_VERSION,
    basis_points,
    damage_bin_index,
    empirical_action_value,
)


def combat_context(*, distance: float, enemy_y: float, enemy_x: float, phase: str) -> str:
    distance_bucket = "close" if distance < 80 else "mid" if distance < 260 else "far"
    altitude = "ground" if enemy_y < 44 else "air"
    corner = "corner" if enemy_x < 150 or enemy_x > 1130 else "field"
    phase_bucket = phase if phase in {"reaction", "recovery", "neutral"} else "danger"
    return f"{distance_bucket}:{altitude}:{corner}:{phase_bucket}"


@dataclass
class ActionOutcomeStats:
    trials: int = 0
    connections: int = 0
    total_damage: int = 0
    total_self_damage: int = 0
    punished_trials: int = 0
    total_spirit_cost: int = 0
    total_commitment: int = 0
    startup_samples: int = 0
    total_startup_to_hit: int = 0
    normalized_samples: int = 0
    total_damage_bp: int = 0
    total_self_damage_bp: int = 0
    total_spirit_cost_bp: int = 0
    self_damage_histogram: list[int] = field(default_factory=lambda: [0] * 8)
    terminal_win_credit: float = 0.0
    terminal_loss_credit: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.connections / max(1, self.trials)

    @property
    def average_damage(self) -> float:
        return self.total_damage / max(1, self.trials)

    @property
    def average_self_damage(self) -> float:
        return self.total_self_damage / max(1, self.trials)

    @property
    def average_startup(self) -> float | None:
        return (
            self.total_startup_to_hit / self.startup_samples
            if self.startup_samples else None
        )


@dataclass
class OffenseEpisode:
    label: str
    context: str
    start_frame: int
    deadline: int
    start_enemy_hp: int
    start_me_hp: int
    start_spirit: int
    max_enemy_hp: int
    max_me_hp: int
    max_spirit: int
    first_hit_frame: int | None = None
    enemy_hp: int = 0
    me_hp: int = 0
    spirit: int = 0


@dataclass
class RecentOutcome:
    label: str
    context: str
    end_frame: int


@dataclass
class RoundOutcomeStats:
    rounds: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_remaining_hp_bp: int = 0
    total_enemy_remaining_hp_bp: int = 0


class ActionOutcomeModel:
    """Online accumulator; selection consumes only compact aggregate rows."""

    def __init__(self) -> None:
        self.table: dict[str, dict[str, ActionOutcomeStats]] = {}
        self.episode: OffenseEpisode | None = None
        self.recent: deque[RecentOutcome] = deque(maxlen=32)
        self.rounds = RoundOutcomeStats()

    def begin(
        self,
        label: str,
        context: str,
        *,
        frame: int,
        commitment: int,
        enemy_hp: int,
        me_hp: int,
        spirit: int,
        max_enemy_hp: int = 10000,
        max_me_hp: int = 10000,
        max_spirit: int = 1000,
    ) -> None:
        if self.episode is not None:
            self.finish(frame)
        self.episode = OffenseEpisode(
            label=label,
            context=context,
            start_frame=frame,
            deadline=frame + max(1, commitment),
            start_enemy_hp=enemy_hp,
            start_me_hp=me_hp,
            start_spirit=spirit,
            max_enemy_hp=max(1, max_enemy_hp),
            max_me_hp=max(1, max_me_hp),
            max_spirit=max(1, max_spirit),
            enemy_hp=enemy_hp,
            me_hp=me_hp,
            spirit=spirit,
        )

    def observe(self, *, frame: int, enemy_hp: int, me_hp: int, spirit: int) -> None:
        episode = self.episode
        if episode is None:
            return
        if enemy_hp < episode.enemy_hp and episode.first_hit_frame is None:
            episode.first_hit_frame = frame
        episode.enemy_hp = enemy_hp
        episode.me_hp = me_hp
        episode.spirit = spirit
        if frame >= episode.deadline:
            self.finish(frame)

    def finish(self, frame: int) -> None:
        episode = self.episode
        self.episode = None
        if episode is None:
            return
        damage = max(0, episode.start_enemy_hp - episode.enemy_hp)
        self_damage = max(0, episode.start_me_hp - episode.me_hp)
        spirit_cost = max(0, episode.start_spirit - episode.spirit)
        commitment = max(1, frame - episode.start_frame)
        for context in (episode.context, "*"):
            stats = self.table.setdefault(context, {}).setdefault(
                episode.label, ActionOutcomeStats()
            )
            stats.trials += 1
            stats.total_damage += damage
            stats.total_self_damage += self_damage
            stats.total_spirit_cost += spirit_cost
            stats.total_commitment += commitment
            damage_bp = basis_points(damage, episode.max_enemy_hp)
            self_damage_bp = basis_points(self_damage, episode.max_me_hp)
            stats.normalized_samples += 1
            stats.total_damage_bp += damage_bp
            stats.total_self_damage_bp += self_damage_bp
            stats.total_spirit_cost_bp += basis_points(
                spirit_cost, episode.max_spirit
            )
            stats.self_damage_histogram[damage_bin_index(self_damage_bp)] += 1
            if damage:
                stats.connections += 1
            if self_damage:
                stats.punished_trials += 1
            if episode.first_hit_frame is not None:
                stats.startup_samples += 1
                stats.total_startup_to_hit += (
                    episode.first_hit_frame - episode.start_frame
                )
        self.recent.append(RecentOutcome(episode.label, episode.context, frame))

    def observe_terminal(
        self,
        *,
        frame: int,
        won: bool | None,
        enemy_hp: int,
        me_hp: int,
        spirit: int,
        max_enemy_hp: int = 10000,
        max_me_hp: int = 10000,
    ) -> None:
        """Assign a small, decayed round result to recently completed options."""
        self.observe(frame=frame, enemy_hp=enemy_hp, me_hp=me_hp, spirit=spirit)
        if self.episode is not None:
            self.finish(frame)
        self.rounds.rounds += 1
        if won is True:
            self.rounds.wins += 1
        elif won is False:
            self.rounds.losses += 1
        else:
            self.rounds.draws += 1
        self.rounds.total_remaining_hp_bp += basis_points(me_hp, max_me_hp)
        self.rounds.total_enemy_remaining_hp_bp += basis_points(
            enemy_hp, max_enemy_hp
        )
        if won is None:
            self.recent.clear()
            return
        for recent in self.recent:
            age = max(0, frame - recent.end_frame)
            if age > 600:
                continue
            credit = 0.85 ** (age / 120.0)
            for context in (recent.context, "*"):
                stats = self.table.get(context, {}).get(recent.label)
                if stats is None:
                    continue
                if won:
                    stats.terminal_win_credit += credit
                else:
                    stats.terminal_loss_credit += credit
        self.recent.clear()

    def stats_for(self, label: str, context: str) -> ActionOutcomeStats | None:
        contextual = self.table.get(context, {}).get(label)
        if contextual is not None and contextual.trials >= 2:
            return contextual
        return self.table.get("*", {}).get(label) or contextual

    def adjustment(self, label: str, context: str, *, explore: bool = False) -> float:
        stats = self.stats_for(label, context)
        if stats is None or stats.trials == 0:
            return 900.0 if explore else 0.0
        observed_utility = empirical_action_value(stats)
        confidence = min(1.0, stats.trials / 6.0)
        return max(-1200.0, min(1200.0, (observed_utility - 900.0) * confidence))

    def export_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            context: {label: asdict(stats) for label, stats in row.items()}
            for context, row in self.table.items()
        }
        state["__reward__"] = {
            "version": REWARD_VERSION,
            "rounds": asdict(self.rounds),
        }
        return state

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        for context, raw_row in state.items():
            if context == "__reward__":
                if isinstance(raw_row, dict) and isinstance(raw_row.get("rounds"), dict):
                    raw_rounds = raw_row["rounds"]
                    self.rounds = RoundOutcomeStats(
                        **{
                            name: int(raw_rounds.get(name, 0))
                            for name in RoundOutcomeStats.__dataclass_fields__
                        }
                    )
                continue
            if not isinstance(raw_row, dict):
                continue
            row = self.table.setdefault(str(context), {})
            for label, raw in raw_row.items():
                if not isinstance(raw, dict):
                    continue
                values: dict[str, object] = {}
                for name in ActionOutcomeStats.__dataclass_fields__:
                    if name == "self_damage_histogram":
                        histogram = raw.get(name, [0] * 8)
                        values[name] = (
                            [max(0, int(value)) for value in histogram[:8]]
                            if isinstance(histogram, list) else [0] * 8
                        )
                        values[name] += [0] * (8 - len(values[name]))
                    elif name in {"terminal_win_credit", "terminal_loss_credit"}:
                        values[name] = float(raw.get(name, 0.0))
                    else:
                        values[name] = int(raw.get(name, 0))
                row[str(label)] = ActionOutcomeStats(**values)

    def metrics(self) -> dict[str, object]:
        return {
            "outcomes": self.export_state(),
            "active": asdict(self.episode) if self.episode else None,
            "rounds": asdict(self.rounds),
            "reward_version": REWARD_VERSION,
        }
