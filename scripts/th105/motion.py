"""Facing-relative fighting-game motion sequences."""

from __future__ import annotations


MOTION_DIGITS: dict[str, tuple[str, ...]] = {
    "236": ("2", "3", "6"),
    "214": ("2", "1", "4"),
    "623": ("6", "2", "3"),
}


def _digit_chord(digit: str, toward: str) -> set[str]:
    if toward not in {"left", "right"}:
        raise ValueError(f"invalid toward direction {toward!r}")
    back = "left" if toward == "right" else "right"
    return {
        "1": {"down", back},
        "2": {"down"},
        "3": {"down", toward},
        "4": {back},
        "5": set(),
        "6": {toward},
    }[digit]


def build_motion_frames(
    motion: str,
    button: str,
    toward: str,
    *,
    direction_frames: int = 2,
    button_frames: int = 3,
    recovery_frames: int = 4,
) -> list[set[str]]:
    if motion not in MOTION_DIGITS:
        raise ValueError(f"unsupported motion {motion!r}")
    if button not in {"z", "x", "c"}:
        raise ValueError(f"unsupported motion button {button!r}")
    if direction_frames <= 0 or button_frames <= 0 or recovery_frames < 0:
        raise ValueError("invalid motion frame counts")

    frames: list[set[str]] = []
    digits = MOTION_DIGITS[motion]
    for digit in digits[:-1]:
        frames.extend([_digit_chord(digit, toward)] * direction_frames)
    final = _digit_chord(digits[-1], toward) | {button}
    frames.extend([final] * button_frames)
    frames.extend([set()] * recovery_frames)
    return [set(chord) for chord in frames]


def build_close_normal_chain(toward: str) -> list[set[str]]:
    """A -> 3A-style bootstrap chain, later gated by cancel metadata."""
    return (
        [{"z"}] * 3
        + [set()] * 3
        + [{"down", toward, "z"}] * 3
        + [set()] * 6
    )


def _pulse(chord: set[str], pressed: int, released: int) -> list[set[str]]:
    return [set(chord) for _ in range(pressed)] + [set() for _ in range(released)]


def build_beginner_ground_combo(toward: str) -> list[set[str]]:
    """Guide-derived AAA > 2B > C > 236B bootstrap timing.

    These conservative spacings are intentionally centralized so live action
    telemetry can tune them without changing the policy/bridge interface.
    """
    if toward not in {"left", "right"}:
        raise ValueError(f"invalid toward direction {toward!r}")
    frames: list[set[str]] = []
    for release_frames in (6, 6, 8):
        frames.extend(_pulse({"z"}, 2, release_frames))
    frames.extend(_pulse({"down", "x"}, 3, 9))
    frames.extend(_pulse({"c"}, 3, 9))
    frames.extend(build_motion_frames("236", "x", toward, recovery_frames=8))
    return frames


def build_dash_frames(direction: str, *, hold_frames: int = 10) -> list[set[str]]:
    """66/44 ground or air dash with a distinct second edge."""
    if direction not in {"left", "right"} or hold_frames <= 0:
        raise ValueError("invalid dash direction or hold")
    return [{direction}] * 2 + [set()] * 2 + [{direction}] * hold_frames + [set()] * 3


def build_jump_frames(
    toward: str,
    direction: str = "neutral",
    *,
    super_jump: bool = False,
    hold_frames: int = 10,
) -> list[set[str]]:
    """7/8/9 jump or 27/28/29 grazing super jump."""
    if toward not in {"left", "right"}:
        raise ValueError(f"invalid toward direction {toward!r}")
    if direction not in {"back", "neutral", "toward"} or hold_frames <= 0:
        raise ValueError("invalid jump direction or hold")
    horizontal = None
    if direction == "toward":
        horizontal = toward
    elif direction == "back":
        horizontal = "left" if toward == "right" else "right"
    jump = {"up"} | ({horizontal} if horizontal else set())
    prefix = [{"down"}] * 2 + [set()] if super_jump else []
    return prefix + [jump] * hold_frames + [set()] * 3


def build_flight_frames(direction: set[str], *, hold_frames: int = 12) -> list[set[str]]:
    """Airborne D+direction flight; physical keyboard A is the D button."""
    if not direction or direction - {"up", "down", "left", "right"}:
        raise ValueError(f"invalid flight direction {sorted(direction)!r}")
    if hold_frames <= 0:
        raise ValueError("invalid flight hold")
    return [set(direction) | {"a"}] * hold_frames + [set()] * 3
