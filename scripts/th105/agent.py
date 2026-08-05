from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .constants import (
    ADDR_CURRENT_ENGINE_SCENE,
    ADDR_COMBINED_MENU_INPUT,
    ADDR_P1_INPUT,
    ADDR_RAW_KEYBOARD,
    ADDR_REQUESTED_ENGINE_SCENE,
    ADDR_SCENE_CONTROLLER,
    TARGET_EXE,
    configured_game_dir,
)
from .input import KEYS, VIRTUAL_KEYS, Keyboard
from .injected_input import InjectedInputBridge, InjectedKeyboard
from .combo import parse_combo, play_combo
from .battle import (
    battle_state_json,
    inspect_frame_boxes,
    read_active_projectiles,
    read_battle_state,
    run_adaptive_fight,
    run_bootstrap_fight,
)
from .menu import (
    CHARACTERS,
    enter_arcade_battle,
    game_mode,
    main_menu_selection,
    reach_main_menu,
    scene_id,
)
from .win32 import ProcessReader, Win32, find_exact_target, verify_reader, wait_exact_target


def target_path(game_dir: Path) -> Path:
    path = game_dir / TARGET_EXE
    if not path.is_file():
        raise FileNotFoundError(f"missing game executable: {path}")
    return path


def open_target(api: Win32, game_dir: Path) -> tuple[ProcessReader, dict[str, object]]:
    pid, _ = find_exact_target(api, target_path(game_dir))
    reader = ProcessReader(api, pid)
    try:
        identity = verify_reader(reader, target_path(game_dir))
    except Exception:
        reader.close()
        raise
    return reader, identity


def launch_target(api: Win32, game_dir: Path, timeout: float) -> tuple[int, dict[str, object]]:
    exe = target_path(game_dir)
    if api.find_pids(TARGET_EXE):
        pid, identity = find_exact_target(api, exe)
        return pid, identity
    pid = api.launch(exe)
    return pid, wait_exact_target(api, exe, pid, timeout)


def probe(args: argparse.Namespace) -> int:
    api = Win32()
    reader, identity = open_target(api, args.game_dir)
    try:
        controller = reader.u32(ADDR_SCENE_CONTROLLER)
        battle = (
            read_battle_state(reader)
            if scene_id(reader) == 5 and reader.u32(0x006E6244)
            else None
        )
        projectile_probe: dict[str, object] | None = None
        if battle is not None:
            p1_projectiles = read_active_projectiles(reader, battle.p1, battle.p2)
            p2_projectiles = read_active_projectiles(reader, battle.p2, battle.p1)
            projectile_probe = {
                "p1": [
                    {
                        **vars(item),
                        "pointer": f"0x{item.pointer:08X}",
                        "frame_data": f"0x{item.frame_data:08X}",
                    }
                    for item in p1_projectiles
                ],
                "p2": [
                    {
                        **vars(item),
                        "pointer": f"0x{item.pointer:08X}",
                        "frame_data": f"0x{item.frame_data:08X}",
                    }
                    for item in p2_projectiles
                ],
                "frame_data": {
                    f"0x{frame_data:08X}": inspect_frame_boxes(reader, frame_data)
                    for frame_data in {
                        item.frame_data for item in (*p1_projectiles, *p2_projectiles)
                    }
                    if frame_data >= 0x10000
                },
                "fighter_frame_data": {
                    "p1": inspect_frame_boxes(reader, battle.p1.frame_data),
                    "p2": inspect_frame_boxes(reader, battle.p2.frame_data),
                },
            }
        state = {
            "scene_id": scene_id(reader),
            "requested_scene_id": reader.u32(ADDR_REQUESTED_ENGINE_SCENE),
            "scene_address": f"0x{ADDR_CURRENT_ENGINE_SCENE:08X}",
            "scene_controller": f"0x{controller:08X}",
            "scene_vtable": f"0x{reader.u32(controller):08X}" if controller else None,
            "input_hook_bytes": reader.read(0x00408218, 5).hex(" "),
            "game_mode": game_mode(reader),
            "main_menu_selection": (
                main_menu_selection(reader) if scene_id(reader) == 2 else None
            ),
            "raw_keyboard_pressed": [
                name for name, key in KEYS.items()
                if reader.read(ADDR_RAW_KEYBOARD + key.dik_code, 1)[0] & 0x80
            ],
            "async_keyboard_pressed": [
                name for name, virtual_key in VIRTUAL_KEYS.items()
                if api.user32.GetAsyncKeyState(virtual_key) & 0x8000
            ],
            "p1_input": {
                "device_index": reader.i8(ADDR_P1_INPUT + 4),
                "bindings": [reader.u32(ADDR_P1_INPUT + offset) for offset in range(8, 56, 4)],
                "counters": [reader.i32(ADDR_P1_INPUT + offset) for offset in range(56, 96, 4)],
                "logical_mask": reader.u16(ADDR_P1_INPUT + 98),
            },
            "menu_input": {
                "counters": [
                    reader.i32(ADDR_COMBINED_MENU_INPUT + offset)
                    for offset in range(56, 96, 4)
                ],
                "source_count": reader.u32(ADDR_COMBINED_MENU_INPUT + 112),
            },
            "battle": battle_state_json(battle) if battle is not None else None,
            "projectiles": projectile_probe,
            "foreground": api.foreground_pid() == reader.pid,
            "windows": [
                {
                    "handle": window,
                    "rect": api.window_rect(window),
                    "title": api.window_text(window),
                    "class": api.window_class(window),
                }
                for window in api.windows_for_pid(reader.pid)
            ],
        }
        print(json.dumps({"identity": identity, "state": state}, ensure_ascii=False))
    finally:
        reader.close()
    return 0


def trace_actions(args: argparse.Namespace) -> int:
    """Sample native fighter transitions without touching input or focus."""
    api = Win32()
    reader, identity = open_target(api, args.game_dir)
    events: list[dict[str, object]] = []
    last: tuple[object, ...] | None = None
    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline and len(events) < args.max_events:
            if scene_id(reader) != 5:
                time.sleep(1.0 / args.sample_hz)
                continue
            state = read_battle_state(reader)
            if args.coarse:
                signature = (
                    state.p1.action_id,
                    state.p1.hp,
                    state.p1.command_mask,
                    state.p2.action_id,
                    state.p2.hp,
                    state.p2.command_mask,
                )
            else:
                signature = (
                    state.p1.action_id,
                    state.p1.action_sequence,
                    state.p1.action_pose,
                    state.p1.action_frame,
                    state.p1.frame_data_flags,
                    state.p1.hp,
                    state.p2.action_id,
                    state.p2.action_sequence,
                    state.p2.action_pose,
                    state.p2.action_frame,
                    state.p2.frame_data_flags,
                    state.p2.hp,
                )
            if signature != last:
                events.append(
                    {
                        "t": args.seconds - max(0.0, deadline - time.perf_counter()),
                        "p1": battle_state_json(state)["p1"],
                        "p2": battle_state_json(state)["p2"],
                    }
                )
                last = signature
            time.sleep(1.0 / args.sample_hz)
    finally:
        reader.close()
    print(json.dumps({"identity": identity, "events": events}, ensure_ascii=False))
    return 0


def tap(args: argparse.Namespace) -> int:
    api = Win32()
    reader, identity = open_target(api, args.game_dir)
    bridge = None
    keyboard = Keyboard(api, reader.pid)
    history: list[dict[str, int | str]] = []
    try:
        if args.backend == "bridge":
            bridge = InjectedInputBridge(api, reader)
            bridge.install()
            keyboard = InjectedKeyboard(api, reader.pid, bridge)
        api.focus(reader.pid)
        keyboard.release_all(require_foreground=True)
        for name in args.keys:
            before = scene_id(reader)
            keyboard.tap(name, args.hold_ms, args.gap_ms)
            after = scene_id(reader)
            history.append({"key": name, "before": before, "after": after})
    finally:
        try:
            keyboard.release_all()
        finally:
            if bridge is not None:
                bridge.close()
            reader.close()
    print(json.dumps({"identity": identity, "history": history}, ensure_ascii=False))
    return 0


def hold(args: argparse.Namespace) -> int:
    api = Win32()
    reader, _identity = open_target(api, args.game_dir)
    bridge = None
    keyboard = Keyboard(api, reader.pid)
    names = set(args.chord.split("+"))
    try:
        if args.backend == "bridge":
            bridge = InjectedInputBridge(api, reader)
            bridge.install()
            keyboard = InjectedKeyboard(api, reader.pid, bridge)
        api.focus(reader.pid)
        keyboard.release_all(require_foreground=True)
        keyboard.hold_chord(names, args.seconds)
    finally:
        try:
            keyboard.release_all()
        finally:
            if bridge is not None:
                bridge.close()
            reader.close()
    return 0


def release(args: argparse.Namespace) -> int:
    api = Win32()
    reader, _identity = open_target(api, args.game_dir)
    try:
        api.focus(reader.pid)
        Keyboard(api, reader.pid).release_all(require_foreground=True)
    finally:
        reader.close()
    return 0


def combo(args: argparse.Namespace) -> int:
    api = Win32()
    reader, _identity = open_target(api, args.game_dir)
    bridge = InjectedInputBridge(api, reader)
    try:
        bridge.install()
        keyboard = InjectedKeyboard(api, reader.pid, bridge)
        api.focus(reader.pid)
        keyboard.release_all(require_foreground=True)
        play_combo(keyboard, parse_combo(args.steps), frame_hz=args.frame_hz)
    finally:
        bridge.release_all()
        bridge.close()
        reader.close()
    return 0


def fight(args: argparse.Namespace) -> int:
    api = Win32()
    reader, identity = open_target(api, args.game_dir)
    bridge = InjectedInputBridge(api, reader)
    result: dict[str, int | float | str]
    try:
        bridge.install()
        keyboard = InjectedKeyboard(api, reader.pid, bridge)
        api.focus(reader.pid, args.timeout)
        keyboard.release_all(require_foreground=True)
        if args.policy == "adaptive":
            result = run_adaptive_fight(
                reader,
                keyboard,
                seconds=args.seconds,
                frame_hz=args.frame_hz,
                policy_path=args.policy_plugin,
                telemetry_path=args.telemetry_path,
            )
        else:
            result = run_bootstrap_fight(
                reader, keyboard, seconds=args.seconds, frame_hz=args.frame_hz
            )
    finally:
        bridge.release_all()
        bridge.close()
        reader.close()
    print(json.dumps({"identity": identity, "fight": result}, ensure_ascii=False))
    return 0


def auto_arcade(args: argparse.Namespace) -> int:
    api = Win32()
    if args.launch:
        pid, identity = launch_target(api, args.game_dir, args.timeout)
    else:
        pid, identity = find_exact_target(api, target_path(args.game_dir))
    reader = ProcessReader(api, pid)
    bridge = InjectedInputBridge(api, reader)
    fight_result: dict[str, object] | None = None
    try:
        bridge.install()
        keyboard = InjectedKeyboard(
            api,
            pid,
            bridge,
            foreground_required=args.foreground_only,
        )
        verify_reader(reader, target_path(args.game_dir))
        api.focus(pid, args.timeout)
        keyboard.release_all(require_foreground=True)
        if scene_id(reader) == 5 and game_mode(reader) == 1:
            history = [{"event": "resume-live-arcade", "scene": 5}]
        else:
            history = reach_main_menu(reader, keyboard, timeout=args.timeout)
            history.extend(
                enter_arcade_battle(
                    reader,
                    keyboard,
                    timeout=args.timeout,
                    p1_character=args.p1_character,
                )
            )
        if args.battle_seconds > 0:
            deadline = time.perf_counter() + args.battle_seconds
            encounters: list[dict[str, object]] = []
            transitions: list[dict[str, int | str]] = []
            last_scene = scene_id(reader)
            while time.perf_counter() < deadline:
                if args.foreground_only and api.foreground_pid() != pid:
                    keyboard.release_all()
                    verify_reader(reader, target_path(args.game_dir))
                    api.focus(pid, args.timeout)
                    transitions.append(
                        {"from": last_scene, "to": last_scene, "event": "focus-reacquired"}
                    )
                    continue
                current = scene_id(reader)
                if current != last_scene:
                    transitions.append({"from": last_scene, "to": current})
                    last_scene = current
                if current == 5:
                    remaining = max(0.05, deadline - time.perf_counter())
                    try:
                        if args.policy == "adaptive":
                            encounter = run_adaptive_fight(
                                reader,
                                keyboard,
                                seconds=remaining,
                                frame_hz=args.frame_hz,
                                policy_path=args.policy_plugin,
                                telemetry_path=args.telemetry_path,
                            )
                        else:
                            encounter = run_bootstrap_fight(
                                reader,
                                keyboard,
                                seconds=remaining,
                                frame_hz=args.frame_hz,
                            )
                    except (OSError, RuntimeError) as exc:
                        if not (
                            isinstance(exc, RuntimeError)
                            and "lost foreground ownership" in str(exc)
                        ):
                            # Scene 5 can briefly outlive its fighter manager at
                            # round/dialogue boundaries. A sparse confirm edge
                            # advances those screens, then the next iteration
                            # reacquires freshly allocated fighters.
                            keyboard.tap("z", hold_ms=35, gap_ms=350)
                            transitions.append(
                                {"from": current, "to": current, "event": "dialogue-confirm"}
                            )
                            continue
                        keyboard.release_all()
                        verify_reader(reader, target_path(args.game_dir))
                        api.focus(pid, args.timeout)
                        transitions.append(
                            {"from": current, "to": current, "event": "focus-reacquired"}
                        )
                        continue
                    encounters.append(dict(encounter))
                    continue
                keyboard.release_all(require_foreground=args.foreground_only)
                if current == 6:
                    # Loading between Arcade opponents; the next manager owns
                    # fresh fighter pointers, so wait and reacquire in scene 5.
                    time.sleep(0.05)
                    continue
                if current == 2:
                    # Arcade loss/completion can return to the main menu. Start
                    # another Easy run with Sakuya without dropping the bridge.
                    history.extend(
                        enter_arcade_battle(
                            reader,
                            keyboard,
                            timeout=args.timeout,
                            p1_character=args.p1_character,
                        )
                    )
                    continue
                keyboard.tap("z", hold_ms=35, gap_ms=500)
                transitions.append(
                    {"from": current, "to": current, "event": "dialogue-confirm"}
                )
            fight_result = {
                "policy": args.policy,
                "encounters": encounters,
                "transitions": transitions,
                "requested_seconds": args.battle_seconds,
            }
        print(
            json.dumps(
                {
                    "identity": identity,
                    "history": history,
                    "final": {
                        "scene_id": scene_id(reader),
                        "game_mode": game_mode(reader),
                        "requested_p1_character": args.p1_character,
                    },
                    "fight": fight_result,
                },
                ensure_ascii=False,
            )
        )
    finally:
        try:
            bridge.release_all()
        finally:
            bridge.close()
            reader.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TH105 exact-target physical controller")
    parser.add_argument("--game-dir", type=Path, default=configured_game_dir())
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="read exact target identity and native scene")
    p.set_defaults(func=probe)

    p = sub.add_parser("trace-actions", help="read-only action/pose/frame transition trace")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--sample-hz", type=float, default=120.0)
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--coarse", action="store_true", help="record action/HP/command changes only")
    p.set_defaults(func=trace_actions)

    p = sub.add_parser("tap", help="send bounded foreground-owned key taps")
    p.add_argument("keys", nargs="+", choices=tuple(KEYS))
    p.add_argument("--hold-ms", type=int, default=65)
    p.add_argument("--gap-ms", type=int, default=220)
    p.add_argument("--backend", choices=("bridge", "sendinput"), default="bridge")
    p.set_defaults(func=tap)

    p = sub.add_parser("hold", help="hold a simultaneous chord for a bounded duration")
    p.add_argument("chord", help="plus-separated keys, for example right+z")
    p.add_argument("--seconds", type=float, default=0.2)
    p.add_argument("--backend", choices=("bridge", "sendinput"), default="bridge")
    p.set_defaults(func=hold)

    p = sub.add_parser("release", help="release every supported injected key")
    p.set_defaults(func=release)

    p = sub.add_parser("combo", help="play frame-timed chord steps through the input bridge")
    p.add_argument("steps", help="for example right@6,right+z@3,neutral@8")
    p.add_argument("--frame-hz", type=float, default=60.0)
    p.set_defaults(func=combo)

    p = sub.add_parser("fight", help="run the visible bootstrap battle loop")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--frame-hz", type=float, default=60.0)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--policy", choices=("adaptive", "bootstrap"), default="adaptive")
    p.add_argument(
        "--policy-plugin",
        type=Path,
        default=Path(__file__).with_name("policies") / "adaptive.py",
        help="hot-reloadable Python policy source",
    )
    p.add_argument(
        "--telemetry-path",
        type=Path,
        default=Path("runtime") / "th105_live.jsonl",
        help="append live combat/action telemetry as JSONL",
    )
    p.set_defaults(func=fight)

    p = sub.add_parser("auto-arcade", help="launch and enter an Easy Arcade battle")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--p1-character", choices=CHARACTERS, default="sakuya")
    p.add_argument("--battle-seconds", type=float, default=0.0)
    p.add_argument("--frame-hz", type=float, default=60.0)
    p.add_argument("--policy", choices=("adaptive", "bootstrap"), default="adaptive")
    p.add_argument(
        "--policy-plugin",
        type=Path,
        default=Path(__file__).with_name("policies") / "adaptive.py",
        help="hot-reloadable Python policy source",
    )
    p.add_argument(
        "--telemetry-path",
        type=Path,
        default=Path("runtime") / "th105_live.jsonl",
        help="append live combat/action telemetry as JSONL",
    )
    p.add_argument(
        "--foreground-only",
        action="store_true",
        help="pause/reacquire unless TH105 owns the foreground (background is default)",
    )
    p.set_defaults(func=auto_arcade)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.game_dir = args.game_dir.resolve()
    if hasattr(args, "policy_plugin"):
        args.policy_plugin = args.policy_plugin.resolve()
    if hasattr(args, "telemetry_path"):
        args.telemetry_path = args.telemetry_path.resolve()
    return int(args.func(args))
