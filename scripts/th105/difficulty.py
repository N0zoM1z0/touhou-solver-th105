"""Small, deterministic CPU-difficulty curriculum for unattended training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .menu import DIFFICULTIES

DEFAULT_CYCLE_ROUND_QUOTAS = (2, 4, 8, 10)


def next_cyclic_difficulty(level: int) -> str:
    """Return the next fixed campaign level, wrapping Lunatic to Easy."""
    if not 0 <= level < len(DIFFICULTIES):
        raise ValueError(f"invalid current difficulty level {level}")
    return DIFFICULTIES[(level + 1) % len(DIFFICULTIES)]


@dataclass
class FixedRoundDifficultyCycle:
    """Rotate only after an exact number of native terminal rounds."""

    level: int
    round_quotas: tuple[int, ...] = DEFAULT_CYCLE_ROUND_QUOTAS
    completed_rounds: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.level < len(DIFFICULTIES):
            raise ValueError(f"invalid initial difficulty level {self.level}")
        if (
            len(self.round_quotas) != len(DIFFICULTIES)
            or any(quota <= 0 for quota in self.round_quotas)
        ):
            raise ValueError(
                "difficulty round quotas must contain one positive value per level"
            )
        if self.completed_rounds < 0:
            raise ValueError("completed rounds cannot be negative")

    @property
    def difficulty(self) -> str:
        return DIFFICULTIES[self.level]

    @property
    def rounds_per_difficulty(self) -> int:
        return self.round_quotas[self.level]

    @property
    def rotation_due(self) -> bool:
        return self.completed_rounds >= self.rounds_per_difficulty

    @property
    def remaining_rounds(self) -> int:
        return max(0, self.rounds_per_difficulty - self.completed_rounds)

    def record(self, *, wins: int = 0, losses: int = 0, draws: int = 0) -> None:
        if min(wins, losses, draws) < 0:
            raise ValueError("round counts cannot be negative")
        self.completed_rounds += wins + losses + draws

    def choose(self) -> str:
        if self.rotation_due:
            self.level = (self.level + 1) % len(DIFFICULTIES)
            self.completed_rounds = 0
        return self.difficulty

    def status(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "level": self.level,
            "completed_rounds": self.completed_rounds,
            "rounds_per_difficulty": self.rounds_per_difficulty,
            "round_quotas": {
                difficulty: self.round_quotas[index]
                for index, difficulty in enumerate(DIFFICULTIES)
            },
            "remaining_rounds": self.remaining_rounds,
            "rotation_due": self.rotation_due,
        }


@dataclass
class DifficultyCurriculum:
    """Promote/demote one step from a bounded window of native round results."""

    level: int
    min_rounds: int = 6
    window_size: int = 12
    promote_rate: float = 2.0 / 3.0
    demote_rate: float = 0.25
    _results: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.level < len(DIFFICULTIES):
            raise ValueError(f"invalid initial difficulty level {self.level}")
        if self.min_rounds <= 0 or self.window_size < self.min_rounds:
            raise ValueError("curriculum window must contain min_rounds")
        self._results = deque(maxlen=self.window_size)

    @property
    def difficulty(self) -> str:
        return DIFFICULTIES[self.level]

    def record(self, *, wins: int = 0, losses: int = 0, draws: int = 0) -> None:
        if min(wins, losses, draws) < 0:
            raise ValueError("round counts cannot be negative")
        self._results.extend([1.0] * wins)
        self._results.extend([0.0] * losses)
        self._results.extend([0.5] * draws)

    def choose(self) -> str:
        """Return the next campaign difficulty, changing at most one level."""
        if len(self._results) < self.min_rounds:
            return self.difficulty
        score = sum(self._results) / len(self._results)
        changed = False
        if score >= self.promote_rate and self.level + 1 < len(DIFFICULTIES):
            self.level += 1
            changed = True
        elif score <= self.demote_rate and self.level > 0:
            self.level -= 1
            changed = True
        if changed:
            # Require fresh evidence at the new speed instead of repeatedly
            # reusing the same Easy/Hard results to jump multiple levels.
            self._results.clear()
        return self.difficulty

    def status(self) -> dict[str, int | float | str]:
        return {
            "difficulty": self.difficulty,
            "level": self.level,
            "window_rounds": len(self._results),
            "window_score": (
                sum(self._results) / len(self._results) if self._results else 0.0
            ),
        }
