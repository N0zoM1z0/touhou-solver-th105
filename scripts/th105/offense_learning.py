"""Bounded sufficient statistics for attacks and combo outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
    first_hit_frame: int | None = None
    enemy_hp: int = 0
    me_hp: int = 0
    spirit: int = 0


class ActionOutcomeModel:
    """Online accumulator; selection consumes only compact aggregate rows."""

    def __init__(self) -> None:
        self.table: dict[str, dict[str, ActionOutcomeStats]] = {}
        self.episode: OffenseEpisode | None = None

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
            if damage:
                stats.connections += 1
            if self_damage:
                stats.punished_trials += 1
            if episode.first_hit_frame is not None:
                stats.startup_samples += 1
                stats.total_startup_to_hit += (
                    episode.first_hit_frame - episode.start_frame
                )

    def stats_for(self, label: str, context: str) -> ActionOutcomeStats | None:
        contextual = self.table.get(context, {}).get(label)
        if contextual is not None and contextual.trials >= 2:
            return contextual
        return self.table.get("*", {}).get(label) or contextual

    def adjustment(self, label: str, context: str, *, explore: bool = False) -> float:
        stats = self.stats_for(label, context)
        if stats is None or stats.trials == 0:
            return 900.0 if explore else 0.0
        observed_utility = (
            stats.average_damage
            - 1.35 * stats.average_self_damage
            - stats.total_spirit_cost / max(1, stats.trials) * 0.20
            - stats.total_commitment / max(1, stats.trials) * 2.0
        )
        confidence = min(1.0, stats.trials / 6.0)
        return max(-1200.0, min(1200.0, (observed_utility - 900.0) * confidence))

    def export_state(self) -> dict[str, object]:
        return {
            context: {label: asdict(stats) for label, stats in row.items()}
            for context, row in self.table.items()
        }

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        for context, raw_row in state.items():
            if not isinstance(raw_row, dict):
                continue
            row = self.table.setdefault(str(context), {})
            for label, raw in raw_row.items():
                if not isinstance(raw, dict):
                    continue
                row[str(label)] = ActionOutcomeStats(
                    **{
                        field: int(raw.get(field, 0))
                        for field in ActionOutcomeStats.__dataclass_fields__
                    }
                )

    def metrics(self) -> dict[str, object]:
        return {
            "outcomes": self.export_state(),
            "active": asdict(self.episode) if self.episode else None,
        }
