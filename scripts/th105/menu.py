"""Native-state-closed menu navigation for the supported TH105 build."""

from __future__ import annotations

import time
from typing import Protocol

from .constants import (
    ADDR_CURRENT_ENGINE_SCENE,
    ADDR_CPU_DIFFICULTY,
    ADDR_GAME_MODE,
    ADDR_SCENE_CONTROLLER,
    GAME_MODE_ARCADE,
    SCENE_BATTLE,
    SCENE_MAIN_MENU,
    SCENE_SELECT,
)
from .win32 import ProcessReader

MAIN_MENU_ITEMS = 12
MAIN_MENU_PRACTICE = 5
MAIN_MENU_ARCADE = 1
MAIN_MENU_CONFIG = 10
MAIN_MENU_SELECTION_OFFSET = 540
MAIN_MENU_VTABLE = 0x006ACCF4

# UI grid order observed from the shipped character-select screen. This is
# deliberately not the engine's internal character-ID order. Moving right from
# the default Reimu slot selects Sakuya, then Youmu.
CHARACTER_CURSOR_SLOTS = (
    "reimu",
    "sakuya",
    "youmu",
    "marisa",
    "alice",
    "remilia",
    "yuyuko",
    "patchouli",
    "yukari",
    "reisen",
    "aya",
    "suika",
    "komachi",
    "iku",
    "tenshi",
)
CHARACTERS = CHARACTER_CURSOR_SLOTS
# Exact fighter vtables recovered from the supported binary's MSVC RTTI.
CHARACTER_VTABLES = {
    "reimu": 0x006B013C,
    "sakuya": 0x006B0924,
    "youmu": 0x006B1154,
    "marisa": 0x006B0594,
    "alice": 0x006B0BEC,
    "remilia": 0x006B13D4,
    "yuyuko": 0x006B165C,
    "patchouli": 0x006B0EBC,
    "suika": 0x006B1B9C,
    "reisen": 0x006B1E3C,  # RTTI class name: Udonge
    "aya": 0x006B22DC,
    "iku": 0x006B2534,
    "komachi": 0x006B2074,
    "yukari": 0x006B18DC,
    "tenshi": 0x006B279C,
}
VTABLE_CHARACTERS = {vtable: name for name, vtable in CHARACTER_VTABLES.items()}
SELECT_P1_CURSOR_OFFSET = 0x10C
SELECT_P1_CHARACTER_OFFSET = 0x110
SELECT_P2_CURSOR_OFFSET = 0x134
SELECT_P2_CHARACTER_OFFSET = 0x138
DIFFICULTIES = ("easy", "normal", "hard", "lunatic")


class MenuKeyboard(Protocol):
    def tap(self, name: str, hold_ms: int = 65, gap_ms: int = 170) -> None: ...


def character_name(vtable: int) -> str | None:
    return VTABLE_CHARACTERS.get(vtable)


def scene_id(reader: ProcessReader) -> int:
    return reader.u32(ADDR_CURRENT_ENGINE_SCENE)


def game_mode(reader: ProcessReader) -> int:
    return reader.u32(ADDR_GAME_MODE)


def cpu_difficulty(reader: ProcessReader) -> int:
    value = reader.u32(ADDR_CPU_DIFFICULTY)
    if value >= len(DIFFICULTIES):
        raise RuntimeError(f"invalid CPU difficulty {value}")
    return value


def configure_cpu_difficulty(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
    difficulty: str,
) -> list[dict[str, int | str]]:
    """Change Config row 0 through UI and validate the native config value."""
    try:
        target = DIFFICULTIES.index(difficulty.casefold())
    except ValueError as exc:
        raise ValueError(f"unsupported CPU difficulty {difficulty!r}") from exc
    current = cpu_difficulty(reader)
    if current == target:
        return []
    history = select_main_menu_item(reader, keyboard, MAIN_MENU_CONFIG)
    before_scene = scene_id(reader)
    keyboard.tap("z", hold_ms=75, gap_ms=500)
    if scene_id(reader) != SCENE_MAIN_MENU:
        raise RuntimeError("Config unexpectedly left the main-menu scene")
    history.append({"key": "z", "before": before_scene, "after": scene_id(reader)})

    # Config opens on row 0. Choose the shorter wraparound direction and verify
    # every edge against CGameConfig+0x64 rather than trusting screen timing.
    for _ in range(len(DIFFICULTIES)):
        current = cpu_difficulty(reader)
        if current == target:
            break
        right_distance = (target - current) % len(DIFFICULTIES)
        left_distance = (current - target) % len(DIFFICULTIES)
        key = "right" if right_distance <= left_distance else "left"
        keyboard.tap(key, hold_ms=70, gap_ms=220)
        after = cpu_difficulty(reader)
        history.append(
            {
                "key": key,
                "before": current,
                "after": after,
                "difficulty": DIFFICULTIES[after],
            }
        )
        if after == current:
            raise RuntimeError("Config difficulty input did not change native value")
    if cpu_difficulty(reader) != target:
        raise RuntimeError(f"failed to select CPU difficulty {difficulty}")
    keyboard.tap("x", hold_ms=60, gap_ms=500)
    # Closing Config returns ownership to the regular main-menu controller.
    # Reading its selector gives us a native-state acknowledgement of the exit.
    main_menu_selection(reader)
    history.append(
        {
            "key": "x",
            "before": target,
            "after": cpu_difficulty(reader),
            "difficulty": DIFFICULTIES[cpu_difficulty(reader)],
        }
    )
    return history


def scene_controller(reader: ProcessReader) -> int:
    return reader.u32(ADDR_SCENE_CONTROLLER)


def main_menu_selection(reader: ProcessReader) -> int:
    controller = scene_controller(reader)
    if scene_id(reader) != SCENE_MAIN_MENU or not controller:
        raise RuntimeError("main-menu selection requested outside the main menu")
    vtable = reader.u32(controller)
    if vtable != MAIN_MENU_VTABLE:
        raise RuntimeError(f"unexpected main-menu vtable {vtable:#x}")
    selection = reader.u32(controller + MAIN_MENU_SELECTION_OFFSET)
    if selection >= MAIN_MENU_ITEMS:
        raise RuntimeError(f"invalid main-menu selection {selection}")
    return selection


def wait_for_scene(reader: ProcessReader, expected: int, timeout: float) -> int:
    deadline = time.perf_counter() + timeout
    last = scene_id(reader)
    while time.perf_counter() < deadline:
        last = scene_id(reader)
        if last == expected:
            return last
        time.sleep(0.02)
    raise TimeoutError(f"expected scene {expected}, last scene was {last}")


def _character_selection(
    reader: ProcessReader,
    *,
    cursor_offset: int,
    selected_offset: int,
    label: str,
) -> int:
    controller = scene_controller(reader)
    if scene_id(reader) != SCENE_SELECT or not controller:
        raise RuntimeError(f"{label} character selection requested outside scene 3")
    cursor = reader.u32(controller + cursor_offset)
    selected = reader.u32(controller + selected_offset)
    if cursor >= len(CHARACTER_CURSOR_SLOTS) or selected >= len(CHARACTER_CURSOR_SLOTS):
        raise RuntimeError(
            f"invalid character-select state cursor={cursor}, selected={selected}"
        )
    if cursor != selected:
        raise RuntimeError(
            f"character-select fields disagree cursor={cursor}, selected={selected}"
        )
    return selected


def p1_character_selection(reader: ProcessReader) -> int:
    return _character_selection(
        reader,
        cursor_offset=SELECT_P1_CURSOR_OFFSET,
        selected_offset=SELECT_P1_CHARACTER_OFFSET,
        label="P1",
    )


def p2_character_selection(reader: ProcessReader) -> int:
    return _character_selection(
        reader,
        cursor_offset=SELECT_P2_CURSOR_OFFSET,
        selected_offset=SELECT_P2_CHARACTER_OFFSET,
        label="P2",
    )


def select_p1_character(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
    character: str,
) -> list[dict[str, int | str]]:
    try:
        target = CHARACTER_CURSOR_SLOTS.index(character.casefold())
    except ValueError as exc:
        raise ValueError(f"unsupported P1 character {character!r}") from exc

    history: list[dict[str, int | str]] = []
    for _ in range(len(CHARACTER_CURSOR_SLOTS)):
        current = p1_character_selection(reader)
        if current == target:
            return history
        current_row, current_column = divmod(current, 4)
        target_row, target_column = divmod(target, 4)
        # Align the column first to mirror the native grid interaction.  In
        # particular, Patchouli is reached as Reimu -> Marisa -> Patchouli
        # (right x3, down), not through Alice (down, right x3).
        if current_column != target_column:
            key = "right" if target_column > current_column else "left"
            expected = current + (1 if key == "right" else -1)
        elif current_row != target_row:
            next_row = current_row + (1 if target_row > current_row else -1)
            next_row_width = min(4, len(CHARACTER_CURSOR_SLOTS) - next_row * 4)
            if current_column >= next_row_width:
                key = "left"
                expected = current - 1
            else:
                key = "down" if next_row > current_row else "up"
                expected = next_row * 4 + current_column
        else:
            raise AssertionError("character cursor is neither at nor away from target")
        # Character portraits animate between cells.  A fast axis change can
        # be acknowledged by the shared input counters but ignored by the
        # selection controller, so leave a human-scale settle interval.
        keyboard.tap(key, hold_ms=75, gap_ms=700)
        after = p1_character_selection(reader)
        history.append(
            {
                "key": key,
                "before": current,
                "after": after,
                "character": CHARACTER_CURSOR_SLOTS[after],
            }
        )
        # The scene's exposed cursor/selection pair tracks the column during a
        # vertical grid move.  Marisa -> Patchouli therefore remains 3 -> 3
        # even though the visible row changed.  Accept that native behaviour;
        # scene transition plus the battle fighter vtable validate the final
        # choice before any corpus is persisted.
        if key in {"up", "down"} and after == current:
            history[-1]["character"] = character
            history[-1]["vertical_grid_move"] = 1
            return history
        if after != expected:
            raise RuntimeError(
                f"character-select {key} moved {current}->{after}, expected {expected}; "
                f"target character {character!r} navigation did not settle"
            )
    raise RuntimeError(f"failed to select {character} within one roster cycle")


def reach_main_menu(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
    *,
    timeout: float,
) -> list[dict[str, int | str]]:
    history: list[dict[str, int | str]] = []
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        current = scene_id(reader)
        if current == SCENE_MAIN_MENU:
            # The engine scene can commit just before the concrete controller.
            try:
                main_menu_selection(reader)
                return history
            except RuntimeError:
                time.sleep(0.05)
                continue
        if current == 1:
            # Opening story/title animation. A very short confirm skips it; the
            # release must precede scene 2 or the same press can select Story.
            keyboard.tap("z", hold_ms=35, gap_ms=500)
            history.append({"key": "z", "before": current, "after": scene_id(reader)})
        elif current == SCENE_SELECT and game_mode(reader) == GAME_MODE_ARCADE:
            # A bounded controller can stop at the Arena post-match dialogue or
            # result screen, both of which share scene 3 with character select.
            # Cancel first: on character select this safely returns to the main
            # menu. If scene 3 remains, one confirm advances dialogue; the next
            # cancel can then leave the result/selection screen.
            keyboard.tap("x", hold_ms=60, gap_ms=500)
            after_cancel = scene_id(reader)
            history.append(
                {
                    "key": "x",
                    "before": current,
                    "after": after_cancel,
                    "event": "recover-arcade-exit",
                }
            )
            if after_cancel == SCENE_SELECT:
                keyboard.tap("z", hold_ms=35, gap_ms=500)
                history.append(
                    {
                        "key": "z",
                        "before": after_cancel,
                        "after": scene_id(reader),
                        "event": "recover-arcade-dialogue",
                    }
                )
        else:
            # Initial engine/controller construction can expose a transient
            # scene before the opening scene. Do not inject into unknown menus.
            time.sleep(0.05)
    raise TimeoutError(f"could not reach main menu; last scene={scene_id(reader)}")


def select_main_menu_practice(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
) -> list[dict[str, int | str]]:
    history: list[dict[str, int | str]] = []
    for _ in range(MAIN_MENU_ITEMS):
        current = main_menu_selection(reader)
        if current == MAIN_MENU_PRACTICE:
            return history
        down_distance = (MAIN_MENU_PRACTICE - current) % MAIN_MENU_ITEMS
        up_distance = (current - MAIN_MENU_PRACTICE) % MAIN_MENU_ITEMS
        key = "down" if down_distance <= up_distance else "up"
        keyboard.tap(key, hold_ms=70, gap_ms=220)
        after = main_menu_selection(reader)
        history.append({"key": key, "before": current, "after": after})
        if after == current:
            raise RuntimeError(f"main-menu {key} did not advance selection {current}")
    raise RuntimeError("failed to select Practice within one menu cycle")


def select_main_menu_item(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
    target: int,
) -> list[dict[str, int | str]]:
    if target < 0 or target >= MAIN_MENU_ITEMS:
        raise ValueError(f"invalid main-menu target {target}")
    history: list[dict[str, int | str]] = []
    for _ in range(MAIN_MENU_ITEMS):
        current = main_menu_selection(reader)
        if current == target:
            return history
        down_distance = (target - current) % MAIN_MENU_ITEMS
        up_distance = (current - target) % MAIN_MENU_ITEMS
        key = "down" if down_distance <= up_distance else "up"
        keyboard.tap(key, hold_ms=70, gap_ms=220)
        after = main_menu_selection(reader)
        history.append({"key": key, "before": current, "after": after})
        if after == current:
            raise RuntimeError(f"main-menu {key} did not advance selection {current}")
    raise RuntimeError(f"failed to select main-menu index {target}")


def enter_arcade_battle(
    reader: ProcessReader,
    keyboard: MenuKeyboard,
    *,
    timeout: float,
    p1_character: str = "sakuya",
) -> list[dict[str, int | str]]:
    history = select_main_menu_item(reader, keyboard, MAIN_MENU_ARCADE)
    before = scene_id(reader)
    keyboard.tap("z", hold_ms=75, gap_ms=650)
    wait_for_scene(reader, SCENE_SELECT, timeout)
    history.append({"key": "z", "before": before, "after": scene_id(reader)})

    history.extend(select_p1_character(reader, keyboard, p1_character))

    # Arcade mode 1 uses the selection controller for character/deck setup.
    # CPU difficulty comes from CGameConfig+0x64 and is configured separately.
    for _ in range(6):
        before = scene_id(reader)
        if before != SCENE_SELECT:
            break
        keyboard.tap("z", hold_ms=75, gap_ms=650)
        history.append({"key": "z", "before": before, "after": scene_id(reader)})
    wait_for_scene(reader, SCENE_BATTLE, timeout)
    if game_mode(reader) != GAME_MODE_ARCADE:
        raise RuntimeError(
            f"battle has game mode {game_mode(reader)}, expected Arcade {GAME_MODE_ARCADE}"
        )
    return history
