"""Online, per-encounter model of opponent action commitment and recovery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionProfile:
    action_id: int
    starts: int = 0
    completions: int = 0
    total_frames: float = 0.0
    impact_samples: int = 0
    first_impact_frame: float = 0.0
    last_impact_frame: float = 0.0
    projectile_samples: int = 0
    first_projectile_frame: float = 0.0
    last_projectile_frame: float = 0.0
    active_box_observations: int = 0
    first_active_box_frame: float = 0.0
    last_active_box_frame: float = 0.0
    maximum_observed_frame: int = 0

    @staticmethod
    def _mean(old: float, count: int, value: float) -> float:
        return value if count == 1 else old + (value - old) / count

    def complete(self, duration: int) -> None:
        self.completions += 1
        self.total_frames = self._mean(
            self.total_frames, self.completions, float(max(1, duration))
        )

    def record_impact(self, elapsed: int) -> None:
        self.impact_samples += 1
        self.first_impact_frame = self._mean(
            self.first_impact_frame, self.impact_samples, float(elapsed)
        )
        self.last_impact_frame = max(self.last_impact_frame, float(elapsed))

    def record_projectile(self, elapsed: int) -> None:
        self.projectile_samples += 1
        self.first_projectile_frame = self._mean(
            self.first_projectile_frame, self.projectile_samples, float(elapsed)
        )
        self.last_projectile_frame = max(self.last_projectile_frame, float(elapsed))

    def record_active_box(self, elapsed: int) -> None:
        value = float(elapsed)
        if self.active_box_observations == 0:
            self.first_active_box_frame = value
        else:
            self.first_active_box_frame = min(self.first_active_box_frame, value)
        self.active_box_observations += 1
        self.last_active_box_frame = max(self.last_active_box_frame, value)

    @property
    def last_threat_frame(self) -> float | None:
        samples = []
        if self.impact_samples:
            samples.append(self.last_impact_frame)
        if self.projectile_samples:
            samples.append(self.last_projectile_frame)
        if self.active_box_observations:
            samples.append(self.last_active_box_frame)
        return max(samples, default=None)

    @property
    def first_threat_frame(self) -> float | None:
        samples = []
        if self.impact_samples:
            samples.append(self.first_impact_frame)
        if self.projectile_samples:
            samples.append(self.first_projectile_frame)
        if self.active_box_observations:
            samples.append(self.first_active_box_frame)
        return min(samples, default=None)


@dataclass(frozen=True)
class ActionAssessment:
    action_id: int
    elapsed: int
    phase: str
    remaining_frames: float
    punish_window: float
    confidence: float
    spell: bool


@dataclass
class _Episode:
    action_id: int
    start_frame: int
    start_projectiles: int


class OpponentActionModel:
    """Learn temporal action envelopes without assuming authoritative source data.

    Damage attribution is intentionally conservative: only HP loss while the
    opponent remains in the same offensive action is recorded. Projectile
    births are detected from count increases and become a second active marker.
    """

    def __init__(self) -> None:
        self.profiles: dict[int, ActionProfile] = {}
        self.episode: _Episode | None = None
        self.last_enemy_action: int | None = None
        self.last_me_hp: int | None = None
        self.last_projectile_count = 0

    def reset_episode(self) -> None:
        """Drop boundary state while retaining learned per-action profiles."""
        self.episode = None
        self.last_enemy_action = None
        self.last_me_hp = None
        self.last_projectile_count = 0

    def seed(self, metrics: dict[str, object]) -> None:
        """Load cumulative profiles previously learned for this character."""
        for raw_action_id, raw in metrics.items():
            if not isinstance(raw, dict):
                continue
            action_id = int(raw_action_id)
            self.profiles[action_id] = ActionProfile(
                action_id=action_id,
                starts=int(raw.get("starts", 0)),
                completions=int(raw.get("completions", 0)),
                total_frames=float(raw.get("mean_total", 0.0)),
                impact_samples=int(raw.get("impact_samples", 0)),
                first_impact_frame=float(raw.get("first_impact", 0.0)),
                last_impact_frame=float(raw.get("last_impact", 0.0)),
                projectile_samples=int(raw.get("projectile_samples", 0)),
                first_projectile_frame=float(raw.get("first_projectile", 0.0)),
                last_projectile_frame=float(raw.get("last_projectile", 0.0)),
                active_box_observations=int(raw.get("active_box_observations", 0)),
                first_active_box_frame=float(raw.get("first_active_box", 0.0)),
                last_active_box_frame=float(raw.get("last_active_box", 0.0)),
            )

    @staticmethod
    def is_offensive(action_id: int) -> bool:
        return 300 <= action_id < 800

    def observe(
        self,
        frame: int,
        *,
        enemy_action: int,
        me_hp: int,
        projectile_count: int,
        active_hitbox: bool = False,
    ) -> ActionAssessment:
        if enemy_action != self.last_enemy_action:
            if self.episode is not None:
                profile = self.profiles[self.episode.action_id]
                profile.complete(frame - self.episode.start_frame)
                self.episode = None
            if self.is_offensive(enemy_action):
                profile = self.profiles.setdefault(
                    enemy_action, ActionProfile(enemy_action)
                )
                profile.starts += 1
                self.episode = _Episode(enemy_action, frame, projectile_count)
            self.last_enemy_action = enemy_action

        if self.episode is not None and self.episode.action_id == enemy_action:
            elapsed = max(0, frame - self.episode.start_frame)
            profile = self.profiles[enemy_action]
            profile.maximum_observed_frame = max(
                profile.maximum_observed_frame, elapsed
            )
            if self.last_me_hp is not None and me_hp < self.last_me_hp:
                profile.record_impact(elapsed)
            if projectile_count > self.last_projectile_count:
                profile.record_projectile(elapsed)
            if active_hitbox:
                profile.record_active_box(elapsed)

        self.last_me_hp = me_hp
        self.last_projectile_count = projectile_count
        return self.assess(frame, enemy_action)

    def assess(self, frame: int, enemy_action: int) -> ActionAssessment:
        spell = 600 <= enemy_action < 800
        if self.episode is None or self.episode.action_id != enemy_action:
            phase = "reaction" if 50 <= enemy_action < 200 else "neutral"
            return ActionAssessment(enemy_action, 0, phase, 0.0, 0.0, 0.0, spell)

        elapsed = max(0, frame - self.episode.start_frame)
        profile = self.profiles[enemy_action]
        estimated_total = (
            profile.total_frames
            if profile.completions
            else float(max(profile.maximum_observed_frame + 12, 36))
        )
        remaining = max(0.0, estimated_total - elapsed)
        last_threat = profile.last_threat_frame
        first_threat = profile.first_threat_frame
        confidence = min(1.0, (profile.completions + profile.impact_samples + profile.projectile_samples) / 4.0)

        if spell:
            phase = "spell-danger"
            punish_window = 0.0
        elif last_threat is None:
            phase = "unknown"
            punish_window = 0.0
        elif elapsed <= last_threat + 2.0:
            phase = "active" if elapsed >= max(0.0, first_threat or 0.0) else "startup"
            punish_window = 0.0
        else:
            phase = "recovery"
            punish_window = remaining
        return ActionAssessment(
            enemy_action,
            elapsed,
            phase,
            remaining,
            punish_window,
            confidence,
            spell,
        )

    def metrics(self) -> dict[str, object]:
        return {
            str(action_id): {
                "starts": profile.starts,
                "completions": profile.completions,
                "mean_total": profile.total_frames,
                "impact_samples": profile.impact_samples,
                "first_impact": profile.first_impact_frame,
                "last_impact": profile.last_impact_frame,
                "projectile_samples": profile.projectile_samples,
                "first_projectile": profile.first_projectile_frame,
                "last_projectile": profile.last_projectile_frame,
                "active_box_observations": profile.active_box_observations,
                "first_active_box": profile.first_active_box_frame,
                "last_active_box": profile.last_active_box_frame,
            }
            for action_id, profile in self.profiles.items()
        }
