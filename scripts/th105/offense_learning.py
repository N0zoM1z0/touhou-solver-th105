"""Bounded sufficient statistics for attacks and combo outcomes."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from .reward import (
    REWARD_VERSION,
    RewardConfig,
    basis_points,
    damage_bin_index,
    empirical_action_value,
)


def combat_context(
    *, distance: float, enemy_y: float, enemy_x: float, phase: str
) -> str:
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
    effectful_trials: int = 0
    action_change_trials: int = 0
    projectile_spawn_trials: int = 0
    total_displacement: int = 0
    delayed_damage_credit_bp: float = 0.0
    delayed_self_damage_credit_bp: float = 0.0
    eligibility_events: int = 0

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
            if self.startup_samples
            else None
        )


def normalize_legacy_outcome_stats(stats: ActionOutcomeStats) -> None:
    """Backfill raw TH105 samples before mixed old/new confidence diverges."""
    legacy_samples = max(0, stats.trials - stats.normalized_samples)
    if legacy_samples <= 0:
        return
    # The supported build uses 10000 HP and 1000 spirit: one raw HP point is
    # one basis point, while one raw spirit point is ten basis points.
    residual_damage = max(0, stats.total_damage - stats.total_damage_bp)
    residual_self_damage = max(0, stats.total_self_damage - stats.total_self_damage_bp)
    residual_spirit_bp = max(
        0, stats.total_spirit_cost * 10 - stats.total_spirit_cost_bp
    )
    observed_nonzero = sum(stats.self_damage_histogram[1:])
    legacy_punished = min(
        legacy_samples, max(0, stats.punished_trials - observed_nonzero)
    )
    stats.self_damage_histogram[0] += legacy_samples - legacy_punished
    if legacy_punished:
        mean_punished_bp = round(residual_self_damage / legacy_punished)
        stats.self_damage_histogram[damage_bin_index(mean_punished_bp)] += (
            legacy_punished
        )
    stats.total_damage_bp += residual_damage
    stats.total_self_damage_bp += residual_self_damage
    stats.total_spirit_cost_bp += residual_spirit_bp
    stats.normalized_samples += legacy_samples


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
    start_me_x: float | None = None
    me_x: float | None = None
    start_action_id: int | None = None
    me_action_id: int | None = None
    start_projectile_count: int | None = None
    max_projectile_count: int | None = None


@dataclass
class RecentOutcome:
    label: str
    context: str
    end_frame: int
    max_enemy_hp: int = 10000
    max_me_hp: int = 10000
    projectile_spawned: bool = False
    effectful: bool = False


@dataclass(frozen=True)
class ActionEstimate:
    utility: float
    effective_trials: float
    population_trials: int


def action_hierarchy(label: str) -> tuple[str, ...]:
    """Return bounded, character-agnostic backoff keys for one action label."""
    family, separator, detail = str(label).partition(":")
    if not separator:
        return (f"@family:{family}",)
    keys = [f"@family:{family}"]
    if family == "motion":
        motion = detail.rsplit(":", 1)[-1]
        keys.append(f"@motion:{motion}")
    elif family == "cancel-discovery":
        keys.append(f"@motion:{detail.rsplit(':', 1)[-1]}")
    elif family == "cancel":
        edge_parts = detail.split("|")
        if len(edge_parts) >= 2:
            keys.append(f"@cancel-input:{edge_parts[1]}")
    elif family in {"grammar-probe", "human-probe"}:
        keys.append(f"@attack-pattern:{detail.rsplit(':', 1)[-1]}")
    return tuple(dict.fromkeys(keys))


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
        self.hierarchy: dict[str, dict[str, ActionOutcomeStats]] = {}
        self.episode: OffenseEpisode | None = None
        self.recent: deque[RecentOutcome] = deque(maxlen=128)
        self.rounds = RoundOutcomeStats()
        self.last_observed_enemy_hp: int | None = None
        self.last_observed_me_hp: int | None = None

    @staticmethod
    def _apply_sample(
        stats: ActionOutcomeStats,
        *,
        damage: int,
        self_damage: int,
        spirit_cost: int,
        commitment: int,
        damage_bp: int,
        self_damage_bp: int,
        spirit_cost_bp: int,
        displacement: float,
        action_changed: bool,
        projectile_spawned: bool,
        effectful: bool,
        startup_to_hit: int | None,
    ) -> None:
        stats.trials += 1
        stats.total_damage += damage
        stats.total_self_damage += self_damage
        stats.total_spirit_cost += spirit_cost
        stats.total_commitment += commitment
        stats.normalized_samples += 1
        stats.total_damage_bp += damage_bp
        stats.total_self_damage_bp += self_damage_bp
        stats.total_spirit_cost_bp += spirit_cost_bp
        stats.self_damage_histogram[damage_bin_index(self_damage_bp)] += 1
        stats.connections += int(damage > 0)
        stats.punished_trials += int(self_damage > 0)
        stats.total_displacement += round(displacement)
        stats.action_change_trials += int(action_changed)
        stats.projectile_spawn_trials += int(projectile_spawned)
        stats.effectful_trials += int(effectful)
        if startup_to_hit is not None:
            stats.startup_samples += 1
            stats.total_startup_to_hit += startup_to_hit

    def _credit_recent_damage(self, *, frame: int, enemy_hp: int, me_hp: int) -> None:
        """Eligibility-trace credit for delayed projectiles and option aftermath."""
        enemy_drop = max(0, (self.last_observed_enemy_hp or enemy_hp) - enemy_hp)
        me_drop = max(0, (self.last_observed_me_hp or me_hp) - me_hp)
        self.last_observed_enemy_hp = enemy_hp
        self.last_observed_me_hp = me_hp
        if not enemy_drop and not me_drop:
            return
        eligible: list[tuple[RecentOutcome, float]] = []
        for recent in self.recent:
            age = max(0, frame - recent.end_frame)
            horizon = 360 if recent.projectile_spawned else 150
            if age > horizon:
                continue
            source_weight = 2.5 if recent.projectile_spawned else 1.0 if recent.effectful else 0.35
            weight = source_weight * (0.90 ** (age / 30.0))
            if weight > 0.01:
                eligible.append((recent, weight))
        total_weight = sum(weight for _recent, weight in eligible)
        if total_weight <= 0.0:
            return
        for recent, weight in eligible:
            share = weight / total_weight
            damage_credit = basis_points(enemy_drop, recent.max_enemy_hp) * share
            self_credit = basis_points(me_drop, recent.max_me_hp) * share
            for context in dict.fromkeys((recent.context, "*")):
                for table, labels in (
                    (self.table, (recent.label,)),
                    (self.hierarchy, action_hierarchy(recent.label)),
                ):
                    row = table.get(context, {})
                    for label in labels:
                        stats = row.get(label)
                        if stats is None:
                            continue
                        stats.delayed_damage_credit_bp += damage_credit
                        stats.delayed_self_damage_credit_bp += self_credit
                        stats.eligibility_events += 1

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
        me_x: float | None = None,
        me_action_id: int | None = None,
        projectile_count: int | None = None,
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
            start_me_x=me_x,
            me_x=me_x,
            start_action_id=me_action_id,
            me_action_id=me_action_id,
            start_projectile_count=projectile_count,
            max_projectile_count=projectile_count,
        )

    def observe(
        self,
        *,
        frame: int,
        enemy_hp: int,
        me_hp: int,
        spirit: int,
        me_x: float | None = None,
        me_action_id: int | None = None,
        projectile_count: int | None = None,
    ) -> None:
        self._credit_recent_damage(frame=frame, enemy_hp=enemy_hp, me_hp=me_hp)
        episode = self.episode
        if episode is None:
            return
        if enemy_hp < episode.enemy_hp and episode.first_hit_frame is None:
            episode.first_hit_frame = frame
        episode.enemy_hp = enemy_hp
        episode.me_hp = me_hp
        episode.spirit = spirit
        if me_x is not None:
            episode.me_x = me_x
        if me_action_id is not None:
            episode.me_action_id = me_action_id
        if projectile_count is not None:
            episode.max_projectile_count = max(
                projectile_count,
                episode.max_projectile_count or 0,
            )
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
        displacement = (
            abs(episode.me_x - episode.start_me_x)
            if episode.me_x is not None and episode.start_me_x is not None
            else 0.0
        )
        action_changed = (
            episode.start_action_id is not None
            and episode.me_action_id is not None
            and episode.me_action_id != episode.start_action_id
            and episode.me_action_id >= 300
        )
        projectile_spawned = (
            episode.start_projectile_count is not None
            and episode.max_projectile_count is not None
            and episode.max_projectile_count > episode.start_projectile_count
        )
        effectful = bool(
            damage or displacement >= 80.0 or action_changed or projectile_spawned
        )
        damage_bp = basis_points(damage, episode.max_enemy_hp)
        self_damage_bp = basis_points(self_damage, episode.max_me_hp)
        spirit_cost_bp = basis_points(spirit_cost, episode.max_spirit)
        startup_to_hit = (
            episode.first_hit_frame - episode.start_frame
            if episode.first_hit_frame is not None
            else None
        )
        # Exact rows preserve the full online context. Hierarchy rows share
        # evidence across action families/motions without changing raw corpus.
        for context in dict.fromkeys((episode.context, "*")):
            stats = self.table.setdefault(context, {}).setdefault(
                episode.label, ActionOutcomeStats()
            )
            sample = dict(
                damage=damage,
                self_damage=self_damage,
                spirit_cost=spirit_cost,
                commitment=commitment,
                damage_bp=damage_bp,
                self_damage_bp=self_damage_bp,
                spirit_cost_bp=spirit_cost_bp,
                displacement=displacement,
                action_changed=action_changed,
                projectile_spawned=projectile_spawned,
                effectful=effectful,
                startup_to_hit=startup_to_hit,
            )
            self._apply_sample(stats, **sample)
            hierarchy_row = self.hierarchy.setdefault(context, {})
            for key in action_hierarchy(episode.label):
                self._apply_sample(
                    hierarchy_row.setdefault(key, ActionOutcomeStats()), **sample
                )
        self.recent.append(
            RecentOutcome(
                episode.label,
                episode.context,
                frame,
                max_enemy_hp=episode.max_enemy_hp,
                max_me_hp=episode.max_me_hp,
                projectile_spawned=projectile_spawned,
                effectful=effectful,
            )
        )

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
        self.rounds.total_enemy_remaining_hp_bp += basis_points(enemy_hp, max_enemy_hp)
        if won is None:
            self.recent.clear()
            self.last_observed_enemy_hp = None
            self.last_observed_me_hp = None
            return
        for recent in self.recent:
            age = max(0, frame - recent.end_frame)
            if age > 600:
                continue
            credit = 0.85 ** (age / 120.0)
            for context in dict.fromkeys((recent.context, "*")):
                for table, labels in (
                    (self.table, (recent.label,)),
                    (self.hierarchy, action_hierarchy(recent.label)),
                ):
                    row = table.get(context, {})
                    for label in labels:
                        stats = row.get(label)
                        if stats is None:
                            continue
                        if won:
                            stats.terminal_win_credit += credit
                        else:
                            stats.terminal_loss_credit += credit
        self.recent.clear()
        self.last_observed_enemy_hp = None
        self.last_observed_me_hp = None

    def stats_for(self, label: str, context: str) -> ActionOutcomeStats | None:
        contextual = self.table.get(context, {}).get(label)
        if contextual is not None and contextual.trials >= 2:
            return contextual
        return self.table.get("*", {}).get(label) or contextual

    def estimate(
        self,
        label: str,
        context: str,
        reward_config: RewardConfig | None = None,
    ) -> ActionEstimate:
        """Shrink sparse exact actions toward motion and family evidence."""
        config = reward_config or RewardConfig()
        sources: list[tuple[ActionOutcomeStats | None, float]] = [
            (self.hierarchy.get("*", {}).get(action_hierarchy(label)[0]), 4.0),
            (self.hierarchy.get(context, {}).get(action_hierarchy(label)[0]), 6.0),
        ]
        for key in action_hierarchy(label)[1:]:
            sources.extend(
                (
                    (self.hierarchy.get("*", {}).get(key), 8.0),
                    (self.hierarchy.get(context, {}).get(key), 10.0),
                )
            )
        sources.extend(
            (
                (self.table.get("*", {}).get(label), 12.0),
                (self.table.get(context, {}).get(label), 24.0),
            )
        )
        weighted = 0.0
        support = 0.0
        for stats, cap in sources:
            if stats is None or stats.trials <= 0:
                continue
            weight = min(cap, float(stats.trials))
            weighted += self.empirical_value(stats, config) * weight
            support += weight
        family = action_hierarchy(label)[0]
        population = int(
            getattr(self.hierarchy.get(context, {}).get(family), "trials", 0)
            + getattr(self.hierarchy.get("*", {}).get(family), "trials", 0)
        )
        return ActionEstimate(
            utility=weighted / support if support else 0.0,
            effective_trials=support,
            population_trials=population,
        )

    @staticmethod
    def empirical_value(stats: ActionOutcomeStats, config: RewardConfig) -> float:
        return empirical_action_value(stats, config)

    def adjustment(
        self,
        label: str,
        context: str,
        *,
        explore: bool = False,
        reward_config: RewardConfig | None = None,
    ) -> float:
        stats = self.stats_for(label, context)
        if stats is None or stats.trials == 0:
            return 900.0 if explore else 0.0
        observed_utility = (
            empirical_action_value(stats, reward_config)
            if reward_config
            else empirical_action_value(stats)
        )
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
        state["__hierarchy__"] = {
            context: {label: asdict(stats) for label, stats in row.items()}
            for context, row in self.hierarchy.items()
        }
        return state

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        for context, raw_row in state.items():
            if context == "__reward__":
                if isinstance(raw_row, dict) and isinstance(
                    raw_row.get("rounds"), dict
                ):
                    raw_rounds = raw_row["rounds"]
                    self.rounds = RoundOutcomeStats(
                        **{
                            name: int(raw_rounds.get(name, 0))
                            for name in RoundOutcomeStats.__dataclass_fields__
                        }
                    )
                continue
            if context == "__hierarchy__":
                self._import_rows(raw_row, self.hierarchy)
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
                            if isinstance(histogram, list)
                            else [0] * 8
                        )
                        values[name] += [0] * (8 - len(values[name]))
                    elif name in {
                        "terminal_win_credit",
                        "terminal_loss_credit",
                        "delayed_damage_credit_bp",
                        "delayed_self_damage_credit_bp",
                    }:
                        values[name] = float(raw.get(name, 0.0))
                    else:
                        values[name] = int(raw.get(name, 0))
                stats = ActionOutcomeStats(**values)
                normalize_legacy_outcome_stats(stats)
                row[str(label)] = stats
        if not self.hierarchy:
            self._rebuild_hierarchy()

    def _import_rows(
        self, raw_state: object, target: dict[str, dict[str, ActionOutcomeStats]]
    ) -> None:
        if not isinstance(raw_state, dict):
            return
        for context, raw_row in raw_state.items():
            if not isinstance(raw_row, dict):
                continue
            row = target.setdefault(str(context), {})
            for label, raw in raw_row.items():
                if not isinstance(raw, dict):
                    continue
                values: dict[str, object] = {}
                for name in ActionOutcomeStats.__dataclass_fields__:
                    if name == "self_damage_histogram":
                        histogram = raw.get(name, [0] * 8)
                        values[name] = (
                            [max(0, int(value)) for value in histogram[:8]]
                            if isinstance(histogram, list)
                            else [0] * 8
                        )
                        values[name] += [0] * (8 - len(values[name]))
                    elif name in {
                        "terminal_win_credit",
                        "terminal_loss_credit",
                        "delayed_damage_credit_bp",
                        "delayed_self_damage_credit_bp",
                    }:
                        values[name] = float(raw.get(name, 0.0))
                    else:
                        values[name] = int(raw.get(name, 0))
                stats = ActionOutcomeStats(**values)
                normalize_legacy_outcome_stats(stats)
                row[str(label)] = stats

    def _rebuild_hierarchy(self) -> None:
        """One-time backward-compatible migration from exact v4 rows."""
        for context, row in self.table.items():
            target = self.hierarchy.setdefault(context, {})
            for label, stats in row.items():
                for key in action_hierarchy(label):
                    aggregate = target.setdefault(key, ActionOutcomeStats())
                    for name in ActionOutcomeStats.__dataclass_fields__:
                        value = getattr(stats, name)
                        if name == "self_damage_histogram":
                            aggregate.self_damage_histogram = [
                                left + right
                                for left, right in zip(
                                    aggregate.self_damage_histogram, value
                                )
                            ]
                        else:
                            setattr(aggregate, name, getattr(aggregate, name) + value)

    def metrics(self) -> dict[str, object]:
        return {
            "outcomes": self.export_state(),
            "active": asdict(self.episode) if self.episode else None,
            "rounds": asdict(self.rounds),
            "reward_version": REWARD_VERSION,
        }


class OptionOutcomeModel(ActionOutcomeModel):
    """Longer-horizon outcome model for safe high-level combat options."""

    @staticmethod
    def empirical_value(stats: ActionOutcomeStats, config: RewardConfig) -> float:
        trials = max(1, stats.normalized_samples or stats.trials)
        damage = stats.total_damage_bp / trials
        self_damage = stats.total_self_damage_bp / trials
        delayed = stats.delayed_damage_credit_bp / max(1, stats.trials)
        delayed_self = stats.delayed_self_damage_credit_bp / max(1, stats.trials)
        terminal = (stats.terminal_win_credit - stats.terminal_loss_credit) / max(
            1, stats.trials
        )
        effect = stats.effectful_trials / max(1, stats.trials)
        return (
            config.damage_weight * (damage + 0.65 * delayed)
            - config.self_damage_weight * (self_damage + 0.65 * delayed_self)
            + config.terminal_weight_bp * terminal
            + 0.35 * config.effect_weight_bp * effect
        )

    def metrics(self) -> dict[str, object]:
        return {
            "outcomes": self.export_state(),
            "active": asdict(self.episode) if self.episode else None,
            "rounds": asdict(self.rounds),
            "reward_version": REWARD_VERSION,
        }
