"""Compact frame-timed chords for menu-independent combat experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .input import KEYS


@dataclass(frozen=True)
class ComboStep:
    keys: frozenset[str]
    frames: int


class ChordKeyboard(Protocol):
    def set_chord(self, names: set[str]) -> None: ...
    def release_all(self, *, require_foreground: bool = False) -> None: ...


def parse_combo(specification: str) -> tuple[ComboStep, ...]:
    steps: list[ComboStep] = []
    for raw_step in specification.split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        chord, separator, raw_frames = raw_step.partition("@")
        if not separator:
            raise ValueError(f"combo step needs @frames: {raw_step!r}")
        frames = int(raw_frames)
        if frames <= 0:
            raise ValueError("combo frames must be positive")
        names = frozenset() if chord in ("-", "neutral") else frozenset(chord.split("+"))
        unknown = names - KEYS.keys()
        if unknown:
            raise ValueError(f"unknown combo keys: {sorted(unknown)}")
        steps.append(ComboStep(names, frames))
    if not steps:
        raise ValueError("combo must contain at least one step")
    return tuple(steps)


def play_combo(
    keyboard: ChordKeyboard,
    steps: tuple[ComboStep, ...],
    *,
    frame_hz: float = 60.0,
) -> None:
    if frame_hz <= 0:
        raise ValueError("frame_hz must be positive")
    deadline = time.perf_counter()
    try:
        for step in steps:
            keyboard.set_chord(set(step.keys))
            deadline += step.frames / frame_hz
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        keyboard.release_all()
