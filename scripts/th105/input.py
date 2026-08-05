"""Foreground-owned TH105 scan-code input with chords and fail-safe release."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

from .win32 import (
    INJECTION_MARKER,
    INPUT,
    INPUT_KEYBOARD,
    INPUT_UNION,
    KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP,
    KEYEVENTF_SCANCODE,
    Win32,
    _win_error,
)


@dataclass(frozen=True)
class ScanKey:
    scan_code: int
    dik_code: int
    extended: bool = False


# Values match the DirectInput DIK codes stored in Profile1P.dat.
KEYS: dict[str, ScanKey] = {
    "escape": ScanKey(0x01, 0x01),
    "q": ScanKey(0x10, 0x10),
    "a": ScanKey(0x1E, 0x1E),
    "s": ScanKey(0x1F, 0x1F),
    "d": ScanKey(0x20, 0x20),
    "z": ScanKey(0x2C, 0x2C),
    "confirm": ScanKey(0x2C, 0x2C),
    "x": ScanKey(0x2D, 0x2D),
    "cancel": ScanKey(0x2D, 0x2D),
    "c": ScanKey(0x2E, 0x2E),
    # DirectInput's extended DIK indices include 0x80; SendInput uses the base
    # scan code plus KEYEVENTF_EXTENDEDKEY instead.
    "up": ScanKey(0x48, 0xC8, True),
    "down": ScanKey(0x50, 0xD0, True),
    "left": ScanKey(0x4B, 0xCB, True),
    "right": ScanKey(0x4D, 0xCD, True),
}

VIRTUAL_KEYS: dict[str, int] = {
    "escape": 0x1B,
    "q": ord("Q"),
    "a": ord("A"),
    "s": ord("S"),
    "d": ord("D"),
    "z": ord("Z"),
    "confirm": ord("Z"),
    "x": ord("X"),
    "cancel": ord("X"),
    "c": ord("C"),
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}


def _item(key: ScanKey, pressed: bool) -> INPUT:
    flags = KEYEVENTF_SCANCODE
    if key.extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if not pressed:
        flags |= KEYEVENTF_KEYUP
    return INPUT(
        INPUT_KEYBOARD,
        INPUT_UNION(ki=KEYBDINPUT(0, key.scan_code, flags, 0, INJECTION_MARKER)),
    )


class Keyboard:
    def __init__(self, api: Win32, pid: int) -> None:
        self.api = api
        self.pid = pid
        self.held: set[str] = set()

    def require_foreground(self) -> None:
        if self.api.foreground_pid() != self.pid:
            raise RuntimeError("TH105 lost foreground ownership")

    def _send(self, transitions: tuple[tuple[str, bool], ...]) -> None:
        if not transitions:
            return
        items = (INPUT * len(transitions))(
            *(_item(KEYS[name], pressed) for name, pressed in transitions)
        )
        sent = self.api.user32.SendInput(len(items), items, ctypes.sizeof(INPUT))
        if sent != len(items):
            raise _win_error(f"SendInput sent {sent}/{len(items)} events")

    def set_chord(self, names: set[str]) -> None:
        unknown = names - KEYS.keys()
        if unknown:
            raise ValueError(f"unknown keys: {sorted(unknown)}")
        self.require_foreground()
        transitions = tuple((name, False) for name in sorted(self.held - names)) + tuple(
            (name, True) for name in sorted(names - self.held)
        )
        self._send(transitions)
        self.held = set(names)

    def tap(self, name: str, hold_ms: int = 65, gap_ms: int = 170) -> None:
        self.set_chord({name})
        try:
            time.sleep(hold_ms / 1000.0)
        finally:
            self.set_chord(set())
        time.sleep(gap_ms / 1000.0)

    def hold_chord(self, names: set[str], seconds: float) -> None:
        self.set_chord(names)
        try:
            time.sleep(seconds)
        finally:
            self.set_chord(set())

    def release_all(self, *, require_foreground: bool = False) -> None:
        if require_foreground:
            self.require_foreground()
        # Release every supported physical scan code, not only Python's held set;
        # this also recovers after an interrupted prior process.
        unique = {key for key in KEYS.values()}
        self._send(tuple((name, False) for name in _canonical_names(unique)))
        self.held.clear()


def _canonical_names(keys: set[ScanKey]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[ScanKey] = set()
    for name, key in KEYS.items():
        if key in keys and key not in seen:
            result.append(name)
            seen.add(key)
    return tuple(result)
