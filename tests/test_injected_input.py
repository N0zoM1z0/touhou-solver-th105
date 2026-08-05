from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.injected_input import (
    CONTROL_OFFSET,
    HOOK_ADDRESS,
    HOOK_ORIGINAL,
    InjectedKeyboard,
    build_stub,
    jump_target,
    rel32,
)


class InjectedInputAssemblyTests(unittest.TestCase):
    def test_rel32_round_trip(self) -> None:
        source = HOOK_ADDRESS
        target = 0x12345000
        displacement = struct.unpack("<i", rel32(source, target))[0]
        self.assertEqual(source + 5 + displacement, target)

    def test_near_jump_target_round_trip(self) -> None:
        source = 0x00408218
        target = 0x12345000
        self.assertEqual(jump_target(source, b"\xE9" + rel32(source, target)), target)

    def test_stub_embeds_control_and_restores_original_instruction(self) -> None:
        cave = 0x10000000
        stub = build_stub(cave)
        self.assertIn(struct.pack("<I", cave + CONTROL_OFFSET), stub)
        self.assertIn(HOOK_ORIGINAL, stub)
        self.assertLess(len(stub), CONTROL_OFFSET)
        displacement = struct.unpack("<i", stub[-4:])[0]
        self.assertEqual(cave + len(stub) + displacement, HOOK_ADDRESS + 5)

    def test_background_keyboard_does_not_require_foreground(self) -> None:
        class Api:
            @staticmethod
            def foreground_pid() -> int:
                return 999

        class Bridge:
            keys: set[str] = set()

            def set_keys(self, names: set[str]) -> None:
                self.keys = set(names)

        bridge = Bridge()
        keyboard = InjectedKeyboard(
            Api(), 123, bridge, foreground_required=False  # type: ignore[arg-type]
        )
        keyboard.set_chord({"z"})
        self.assertEqual(bridge.keys, {"z"})

        foreground_keyboard = InjectedKeyboard(
            Api(), 123, bridge, foreground_required=True  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(RuntimeError, "lost foreground"):
            foreground_keyboard.set_chord({"z"})


if __name__ == "__main__":
    unittest.main()
