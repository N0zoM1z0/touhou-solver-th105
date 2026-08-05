"""Online projectile-envelope calibration from observed combat impacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ProjectileSnapshot:
    pointer: int
    action_id: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    acceleration_x: float
    acceleration_y: float

    @classmethod
    def from_state(cls, projectile: Any) -> "ProjectileSnapshot":
        return cls(
            int(projectile.pointer),
            int(projectile.action_id),
            float(projectile.x),
            float(projectile.y),
            float(projectile.velocity_x),
            float(projectile.velocity_y),
            float(getattr(projectile, "acceleration_x", 0.0)),
            float(getattr(projectile, "acceleration_y", 0.0)),
        )

    def projected(self, frames: int = 1) -> "ProjectileSnapshot":
        elapsed = max(0, int(frames))
        return ProjectileSnapshot(
            self.pointer,
            self.action_id,
            self.x + self.velocity_x * elapsed
            + 0.5 * self.acceleration_x * elapsed * elapsed,
            self.y + self.velocity_y * elapsed
            + 0.5 * self.acceleration_y * elapsed * elapsed,
            self.velocity_x + self.acceleration_x * elapsed,
            self.velocity_y + self.acceleration_y * elapsed,
            self.acceleration_x,
            self.acceleration_y,
        )


@dataclass(frozen=True)
class ProjectileImpact:
    action_id: int
    pointer: int
    inferred_extent: float
    disappeared: bool
    gap: float


class ProjectileEnvelopeModel:
    """Learns conservative square hit envelopes, keyed by projectile action.

    TH105 normally destroys a projectile on the same tick that it hits.  The
    previous frame therefore matters more than the post-damage object list.
    We project disappeared objects forward one tick before attributing impact.
    """

    def __init__(self, default_extent: float = 32.0, maximum_extent: float = 80.0) -> None:
        self.default_extent = float(default_extent)
        self.maximum_extent = float(maximum_extent)
        self.extents: dict[int, float] = {}
        self.samples: dict[int, int] = {}
        self.previous: dict[int, ProjectileSnapshot] = {}
        self.recent: dict[int, tuple[ProjectileSnapshot, int]] = {}
        self.tick = 0
        self.last_impact: ProjectileImpact | None = None

    def extent_for(self, action_id: int) -> float:
        return self.extents.get(int(action_id), self.default_extent)

    def observe(
        self,
        projectiles: Iterable[Any],
        *,
        player_x: float,
        player_y: float,
        player_half_width: float,
        player_half_height: float,
        took_damage: bool,
        first_contact: bool,
    ) -> ProjectileImpact | None:
        self.tick += 1
        current = {
            snapshot.pointer: snapshot
            for snapshot in map(ProjectileSnapshot.from_state, projectiles)
        }
        impact: ProjectileImpact | None = None
        if took_damage and first_contact:
            candidates: list[tuple[ProjectileSnapshot, bool]] = [
                (snapshot, False) for snapshot in current.values()
            ]
            candidates.extend(
                (snapshot.projected(self.tick - last_seen), True)
                for pointer, (snapshot, last_seen) in self.recent.items()
                if pointer not in current and self.tick - last_seen <= 3
            )
            ranked: list[tuple[float, float, ProjectileSnapshot, bool]] = []
            for snapshot, disappeared in candidates:
                gap_x = max(0.0, abs(snapshot.x - player_x) - player_half_width)
                gap_y = max(0.0, abs(snapshot.y - player_y) - player_half_height)
                required_extent = max(gap_x, gap_y)
                # Only nearby objects can explain the impact. This rejects
                # incidental bullets while the opponent lands a melee hit.
                if required_extent <= self.maximum_extent:
                    ranked.append(
                        (
                            max(gap_x, gap_y),
                            gap_x * gap_x + gap_y * gap_y,
                            snapshot,
                            disappeared,
                        )
                    )
            if ranked:
                gap, _distance2, snapshot, disappeared = min(ranked)
                learned = min(
                    self.maximum_extent,
                    max(self.default_extent, gap + 8.0),
                )
                action_id = snapshot.action_id
                self.extents[action_id] = max(self.extent_for(action_id), learned)
                self.samples[action_id] = self.samples.get(action_id, 0) + 1
                impact = ProjectileImpact(
                    action_id, snapshot.pointer, learned, disappeared, gap
                )
                self.last_impact = impact
        self.recent = {
            pointer: (snapshot, last_seen)
            for pointer, (snapshot, last_seen) in self.recent.items()
            if self.tick - last_seen < 3 and pointer not in current
        }
        self.recent.update(
            {pointer: (snapshot, self.tick) for pointer, snapshot in current.items()}
        )
        self.previous = current
        return impact

    def export_state(self) -> dict[str, object]:
        return {
            "extents": {str(key): value for key, value in self.extents.items()},
            "samples": {str(key): value for key, value in self.samples.items()},
        }

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        extents = state.get("extents", {})
        samples = state.get("samples", {})
        if isinstance(extents, dict):
            self.extents.update(
                {
                    int(key): min(
                        self.maximum_extent,
                        max(self.default_extent, float(value)),
                    )
                    for key, value in extents.items()
                }
            )
        if isinstance(samples, dict):
            self.samples.update({int(key): int(value) for key, value in samples.items()})

    def metrics(self) -> dict[str, object]:
        return {
            "default_extent": self.default_extent,
            "learned_extents": {str(key): value for key, value in self.extents.items()},
            "samples": {str(key): value for key, value in self.samples.items()},
            "tracked": len(self.previous),
            "recent_missing": sum(
                1 for pointer in self.recent if pointer not in self.previous
            ),
            "last_impact": vars(self.last_impact) if self.last_impact else None,
        }
