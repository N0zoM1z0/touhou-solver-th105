from __future__ import annotations

import os
from pathlib import Path

TARGET_EXE = "th105c.exe"
EXPECTED_EXE_SHA256 = "49c23d9467b9927ba687ed2b873c4bc2d2f39ddadc9f55051ccf10172c0b7c11"
DEFAULT_GAME_DIR = Path(
    r"D:\Entertainment\Game\Touhou\[th105] 东方绯想天 (汉化版+日文版)"
)

# Engine scene transition roots, observed in the shipped 1.06a executable.
# commit_pending_engine_scene (0x00407F80) assigns requested -> current after
# swapping in the pending scene controller.
ADDR_CURRENT_ENGINE_SCENE = 0x006ECE7C
ADDR_REQUESTED_ENGINE_SCENE = 0x006ECE78
ADDR_SCENE_CONTROLLER = 0x006ECE44

# DirectInput keyboard buffer written by the main loop's GetDeviceState call.
ADDR_RAW_KEYBOARD = 0x006ECFF0
ADDR_P1_INPUT = 0x006E6300
ADDR_COMBINED_MENU_INPUT = 0x006E7520
ADDR_GAME_MODE = 0x006E62EC
ADDR_BATTLE_MANAGER = 0x006E6244

# CBattleManagerBase owns the two live fighter pointers at +0x0C/+0x10.
BATTLE_MANAGER_P1_OFFSET = 0x0C
BATTLE_MANAGER_P2_OFFSET = 0x10

SCENE_MAIN_MENU = 2
SCENE_SELECT = 3
SCENE_BATTLE = 5
SCENE_LOADING = 6
GAME_MODE_PRACTICE = 8
GAME_MODE_ARCADE = 1


def configured_game_dir() -> Path:
    return Path(os.environ.get("TH105_GAME_DIR", str(DEFAULT_GAME_DIR)))
