from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.constants import (
    ADDR_CURRENT_ENGINE_SCENE,
    ADDR_CPU_DIFFICULTY,
    ADDR_GAME_MODE,
    ADDR_RAW_KEYBOARD,
    EXPECTED_EXE_SHA256,
    GAME_MODE_ARCADE,
    GAME_MODE_PRACTICE,
    SCENE_BATTLE,
    SCENE_MAIN_MENU,
    SCENE_SELECT,
)
from th105.menu import (
    CHARACTER_CURSOR_SLOTS,
    CHARACTER_VTABLES,
    SELECT_P1_CHARACTER_OFFSET,
    SELECT_P1_CURSOR_OFFSET,
    SELECT_P2_CHARACTER_OFFSET,
    SELECT_P2_CURSOR_OFFSET,
    character_name,
    configure_cpu_difficulty,
)


class DifficultyReader:
    def __init__(self, difficulty: int) -> None:
        self.difficulty = difficulty

    def u32(self, address: int) -> int:
        if address == ADDR_CPU_DIFFICULTY:
            return self.difficulty
        raise AssertionError(f"unexpected read {address:#x}")


class DifficultyKeyboard:
    def __init__(self, reader: DifficultyReader) -> None:
        self.reader = reader
        self.taps: list[str] = []

    def tap(self, name: str, hold_ms: int = 65, gap_ms: int = 170) -> None:
        self.taps.append(name)
        if name == "right":
            self.reader.difficulty = (self.reader.difficulty + 1) % 4
        elif name == "left":
            self.reader.difficulty = (self.reader.difficulty - 1) % 4


class NativeContractTests(unittest.TestCase):
    def test_supported_binary_contract_is_explicit(self) -> None:
        self.assertEqual(len(EXPECTED_EXE_SHA256), 64)
        self.assertEqual(ADDR_CURRENT_ENGINE_SCENE, 0x006ECE7C)
        self.assertEqual(ADDR_RAW_KEYBOARD, 0x006ECFF0)
        self.assertEqual(ADDR_GAME_MODE, 0x006E62EC)
        self.assertEqual((SCENE_MAIN_MENU, SCENE_SELECT, SCENE_BATTLE), (2, 3, 5))
        self.assertEqual(GAME_MODE_PRACTICE, 8)

    def test_sakuya_roster_contract(self) -> None:
        self.assertEqual(CHARACTER_CURSOR_SLOTS[0:3], ("reimu", "sakuya", "youmu"))
        self.assertEqual(SELECT_P1_CURSOR_OFFSET, 0x10C)
        self.assertEqual(SELECT_P1_CHARACTER_OFFSET, 0x110)
        self.assertEqual(SELECT_P2_CURSOR_OFFSET, 0x134)
        self.assertEqual(SELECT_P2_CHARACTER_OFFSET, 0x138)
        self.assertEqual(GAME_MODE_ARCADE, 1)
        self.assertEqual(CHARACTER_VTABLES["sakuya"], 0x006B0924)
        self.assertEqual(character_name(0x006B1E3C), "reisen")
        self.assertIsNone(character_name(0xDEADBEEF))

    def test_configures_difficulty_by_shortest_ui_path(self) -> None:
        reader = DifficultyReader(0)
        keyboard = DifficultyKeyboard(reader)
        with (
            patch("th105.menu.select_main_menu_item", return_value=[]),
            patch("th105.menu.scene_id", return_value=SCENE_MAIN_MENU),
            patch("th105.menu.main_menu_selection", return_value=10),
        ):
            history = configure_cpu_difficulty(reader, keyboard, "lunatic")
        self.assertEqual(reader.difficulty, 3)
        self.assertEqual(keyboard.taps, ["z", "left", "x"])
        self.assertEqual(history[-2]["difficulty"], "lunatic")


if __name__ == "__main__":
    unittest.main()
