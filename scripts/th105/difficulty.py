"""Small, deterministic CPU-difficulty curriculum for unattended training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .menu import DIFFICULTIES


def next_cyclic_difficulty(level: int) -> str:
    """Return the next fixed campaign level, wrapping Lunatic to Easy."""
    if not 0 <= level < len(DIFFICULTIES):
        raise ValueError(f"invalid current difficulty level {level}")
    return DIFFICULTIES[(level + 1) % len(DIFFICULTIES)]


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
