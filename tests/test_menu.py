from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.constants import (
    ADDR_CURRENT_ENGINE_SCENE,
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
    character_name,
)


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
        self.assertEqual(GAME_MODE_ARCADE, 1)
        self.assertEqual(CHARACTER_VTABLES["sakuya"], 0x006B0924)
        self.assertEqual(character_name(0x006B1E3C), "reisen")
        self.assertIsNone(character_name(0xDEADBEEF))


if __name__ == "__main__":
    unittest.main()
