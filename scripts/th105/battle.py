"""Battle-control loops, from visible bootstrap behavior toward state policy."""

from __future__ import annotations

import time
import math
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from .combo import ComboStep, parse_combo, play_combo
from .constants import (
    ADDR_BATTLE_MANAGER,
    BATTLE_MANAGER_P1_OFFSET,
    BATTLE_MANAGER_P2_OFFSET,
    SCENE_BATTLE,
)
from .menu import character_name, scene_id
from .model_compiler import compile_knowledge_file
from .knowledge import (
    attack_geometry_for,
    cancel_graph_for,
    defense_model_for,
    offense_model_for,
    persist_character_models,
    profiles_for,
    projectile_model_for,
)
from .policy_api import PolicyObservation
from .policy_loader import HotReloadPolicy
from .telemetry import BoundedJsonlWriter
from .human_learning import compile_human_file, compiled_demonstrations_for
from .win32 import ProcessReader


class BattleKeyboard(Protocol):
    def set_chord(self, names: set[str], *, require_foreground: bool = True) -> None: ...


# A deliberately conservative visible bootstrap. It keeps Sakuya moving and
# exercising melee/projectile inputs while telemetry fields are being promoted.
# The guard portions are both standing and crouching back for a left-side spawn.
BOOTSTRAP_CYCLE: tuple[ComboStep, ...] = parse_combo(
    "right@18,neutral@3,z@3,neutral@8,x@3,neutral@10,"
    "left@14,down+left@12,neutral@5,right@10,z@3,neutral@9"
)

FIGHTER_POSITION_X = 0xEC
FIGHTER_POSITION_Y = 0xF0
FIGHTER_VELOCITY_X = 0xF4
FIGHTER_VELOCITY_Y = 0xF8
FIGHTER_FACING = 0x104
FIGHTER_ACTION_ID = 0x13C
FIGHTER_ACTION_SEQUENCE = 0x13E
FIGHTER_ACTION_POSE = 0x140
FIGHTER_ACTION_FRAME = 0x142
FIGHTER_CURRENT_FRAME_DATA = 0x158
FIGHTER_HP = 0x174
FIGHTER_MAX_HP = 0x176
FIGHTER_COLLISION_STATE = 0x180
FIGHTER_SPIRIT = 0x482
FIGHTER_OBJECT_MANAGER = 0x658
FIGHTER_COMMAND_MASK = 0x728

FRAME_DATA_FLAGS = 0x4C
FRAME_DATA_ATTACK_FLAGS = 0x50
FRAME_DATA_BODY_BOX = 0x54
FRAME_DATA_BOX_VECTOR_A = 0x5C
FRAME_DATA_BOX_VECTOR_B = 0x6C

OBJECT_SELF = 0x168
OBJECT_TARGET = 0x170
OBJECT_MANAGER_ACTIVE_SENTINEL = 0x5C
OBJECT_MANAGER_ACTIVE_COUNT = 0x60


@dataclass(frozen=True)
class FighterState:
    pointer: int
    vtable: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    facing: int
    action_id: int
    action_sequence: int
    action_pose: int
    action_frame: int
    frame_data_flags: int
    attack_flags: int
    hp: int
    max_hp: int
    collision_state: int
    spirit: int
    command_mask: int
    frame_data: int
    body_box: tuple[int, int, int, int] | None
    hurt_boxes: tuple[tuple[int, int, int, int], ...]
    attack_boxes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class BattleState:
    manager: int
    p1: FighterState
    p2: FighterState


@dataclass
class TerminalRoundTracker:
    """Deduplicate native round results across rebuilt battle loops.

    Projectile/object lists can be transiently invalid while scene 5 remains
    active. The outer controller then starts a fresh ``run_adaptive_fight``;
    this shared tracker only rearms after observing both fighters alive.
    """

    armed: bool = False
    observed_live_round: bool = False
    sequence: int = 0

    def observe(self, *, me_hp: int, enemy_hp: int) -> int | None:
        if me_hp > 0 and enemy_hp > 0:
            self.armed = True
            self.observed_live_round = True
            return None
        if self.armed and self.observed_live_round and (me_hp <= 0 or enemy_hp <= 0):
            self.armed = False
            self.observed_live_round = False
            self.sequence += 1
            return self.sequence
        return None


def cold_start_offense_prior(state: object) -> dict[str, object]:
    """Reuse action evidence without attributing old rounds to a new context."""
    if not isinstance(state, dict):
        return {}
    cloned = copy.deepcopy(state)
    reward = cloned.get("__reward__")
    if isinstance(reward, dict):
        reward["rounds"] = {
            "rounds": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_remaining_hp_bp": 0,
            "total_enemy_remaining_hp_bp": 0,
        }
    return cloned


@dataclass(frozen=True)
class ProjectileState:
    pointer: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    acceleration_x: float
    acceleration_y: float
    facing: int
    action_id: int
    frame_data: int
    attack_flags: int
    body_box: tuple[int, int, int, int] | None
    hurt_boxes: tuple[tuple[int, int, int, int], ...]
    attack_boxes: tuple[tuple[int, int, int, int], ...]


def _read_frame_box(
    reader: ProcessReader, frame_data: int, field_offset: int
) -> tuple[int, int, int, int] | None:
    if frame_data < 0x10000:
        return None
    try:
        pointer = reader.u32(frame_data + field_offset)
        if pointer < 0x10000:
            return None
        box = tuple(reader.i32(pointer + offset) for offset in range(0, 16, 4))
    except OSError:
        return None
    left, top, right, bottom = box
    if left > right or top > bottom or any(abs(value) > 4096 for value in box):
        return None
    return box


def read_frame_box_vector(
    reader: ProcessReader, frame_data: int, field_offset: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Read one bounded std::vector-like array of four-int local rectangles."""
    if frame_data < 0x10000:
        return ()
    try:
        begin = reader.u32(frame_data + field_offset)
        end = reader.u32(frame_data + field_offset + 4)
        capacity = reader.u32(frame_data + field_offset + 8)
    except OSError:
        return ()
    if begin < 0x10000 or end < begin or capacity < end:
        return ()
    byte_count = end - begin
    if byte_count % 16 or byte_count > 16 * 32:
        return ()
    boxes: list[tuple[int, int, int, int]] = []
    try:
        for pointer in range(begin, end, 16):
            box = tuple(reader.i32(pointer + offset) for offset in range(0, 16, 4))
            left, top, right, bottom = box
            if left > right or top > bottom or any(abs(value) > 4096 for value in box):
                return ()
            boxes.append(box)
    except OSError:
        return ()
    return tuple(boxes)


def local_box_to_world(
    box: tuple[int, int, int, int],
    *,
    x: float,
    y: float,
    facing: int,
) -> tuple[float, float, float, float]:
    """Transform a local frame rectangle into game-coordinate world bounds.

    TH105 stores local Y upward as negative while fighter/projectile positions
    use positive Y above the floor.  Facing 1 preserves local X; the opposite
    facing mirrors it around the object origin (matching 0x46ACD0).
    """
    left, top, right, bottom = box
    if facing == 1:
        x1, x2 = x + left, x + right
    else:
        x1, x2 = x - right, x - left
    y1, y2 = y - bottom, y - top
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def boxes_world_envelope(
    boxes: tuple[tuple[int, int, int, int], ...],
    *,
    x: float,
    y: float,
    facing: int,
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    world = tuple(local_box_to_world(box, x=x, y=y, facing=facing) for box in boxes)
    return (
        min(box[0] for box in world),
        min(box[1] for box in world),
        max(box[2] for box in world),
        max(box[3] for box in world),
    )


def inspect_frame_boxes(reader: ProcessReader, frame_data: int) -> dict[str, object]:
    """Bounded diagnostic scan for nearby frame-data rectangle pointers."""
    if frame_data < 0x10000:
        return {"dwords": {}, "candidate_boxes": {}}
    dwords: dict[str, str] = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for offset in range(0x40, 0x84, 4):
        value = reader.u32(frame_data + offset)
        key = f"0x{offset:02X}"
        dwords[key] = f"0x{value:08X}"
        box = _read_frame_box(reader, frame_data, offset)
        if box is not None:
            boxes[key] = box
    return {
        "dwords": dwords,
        "candidate_boxes": boxes,
        "vector_a": read_frame_box_vector(reader, frame_data, FRAME_DATA_BOX_VECTOR_A),
        "vector_b": read_frame_box_vector(reader, frame_data, FRAME_DATA_BOX_VECTOR_B),
    }


def _read_fighter(reader: ProcessReader, pointer: int) -> FighterState:
    if pointer < 0x10000:
        raise RuntimeError(f"invalid fighter pointer {pointer:#x}")
    max_hp = reader.u16(pointer + FIGHTER_MAX_HP)
    if max_hp == 0 or max_hp > 30000:
        raise RuntimeError(f"invalid fighter max HP {max_hp} at {pointer:#x}")
    frame_data = reader.u32(pointer + FIGHTER_CURRENT_FRAME_DATA)
    frame_data_flags = reader.u32(frame_data + FRAME_DATA_FLAGS) if frame_data >= 0x10000 else 0
    attack_flags = reader.u32(frame_data + FRAME_DATA_ATTACK_FLAGS) if frame_data >= 0x10000 else 0
    return FighterState(
        pointer=pointer,
        vtable=reader.u32(pointer),
        x=reader.f32(pointer + FIGHTER_POSITION_X),
        y=reader.f32(pointer + FIGHTER_POSITION_Y),
        velocity_x=reader.f32(pointer + FIGHTER_VELOCITY_X),
        velocity_y=reader.f32(pointer + FIGHTER_VELOCITY_Y),
        facing=reader.i8(pointer + FIGHTER_FACING),
        action_id=reader.u16(pointer + FIGHTER_ACTION_ID),
        action_sequence=reader.u16(pointer + FIGHTER_ACTION_SEQUENCE),
        action_pose=reader.u16(pointer + FIGHTER_ACTION_POSE),
        action_frame=reader.u16(pointer + FIGHTER_ACTION_FRAME),
        frame_data_flags=frame_data_flags,
        attack_flags=attack_flags,
        hp=reader.u16(pointer + FIGHTER_HP),
        max_hp=max_hp,
        collision_state=reader.u32(pointer + FIGHTER_COLLISION_STATE),
        spirit=reader.i16(pointer + FIGHTER_SPIRIT),
        command_mask=reader.u32(pointer + FIGHTER_COMMAND_MASK),
        frame_data=frame_data,
        body_box=_read_frame_box(reader, frame_data, FRAME_DATA_BODY_BOX),
        hurt_boxes=read_frame_box_vector(reader, frame_data, FRAME_DATA_BOX_VECTOR_A),
        attack_boxes=read_frame_box_vector(reader, frame_data, FRAME_DATA_BOX_VECTOR_B),
    )


def read_battle_state(reader: ProcessReader) -> BattleState:
    if scene_id(reader) != SCENE_BATTLE:
        raise RuntimeError(f"battle state requested outside scene {scene_id(reader)}")
    manager = reader.u32(ADDR_BATTLE_MANAGER)
    if manager < 0x10000:
        raise RuntimeError(f"invalid battle manager {manager:#x}")
    return BattleState(
        manager=manager,
        p1=_read_fighter(reader, reader.u32(manager + BATTLE_MANAGER_P1_OFFSET)),
        p2=_read_fighter(reader, reader.u32(manager + BATTLE_MANAGER_P2_OFFSET)),
    )


def battle_state_json(state: BattleState) -> dict[str, object]:
    return {
        "manager": f"0x{state.manager:08X}",
        "p1": {
            **asdict(state.p1),
            "pointer": f"0x{state.p1.pointer:08X}",
            "vtable": f"0x{state.p1.vtable:08X}",
            "character": character_name(state.p1.vtable),
        },
        "p2": {
            **asdict(state.p2),
            "pointer": f"0x{state.p2.pointer:08X}",
            "vtable": f"0x{state.p2.vtable:08X}",
            "character": character_name(state.p2.vtable),
        },
    }


def read_active_projectiles(
    reader: ProcessReader,
    owner: FighterState,
    target: FighterState,
) -> tuple[ProjectileState, ...]:
    manager = reader.u32(owner.pointer + FIGHTER_OBJECT_MANAGER)
    sentinel = reader.u32(manager + OBJECT_MANAGER_ACTIVE_SENTINEL)
    count = reader.u32(manager + OBJECT_MANAGER_ACTIVE_COUNT)
    if not sentinel or count > 256:
        raise RuntimeError(
            f"invalid object-manager active list sentinel={sentinel:#x}, count={count}"
        )
    node = reader.u32(sentinel)
    projectiles: list[ProjectileState] = []
    for _ in range(count):
        if node == sentinel:
            break
        obj = reader.u32(node + 8)
        action_id = reader.u16(obj + FIGHTER_ACTION_ID)
        frame_data = reader.u32(obj + FIGHTER_CURRENT_FRAME_DATA)
        x = reader.f32(obj + FIGHTER_POSITION_X)
        y = reader.f32(obj + FIGHTER_POSITION_Y)
        if (
            800 <= action_id < 1000
            and reader.u32(obj + OBJECT_SELF) == owner.pointer
            and reader.u32(obj + OBJECT_TARGET) == target.pointer
            and math.isfinite(x)
            and math.isfinite(y)
        ):
            projectiles.append(
                ProjectileState(
                    pointer=obj,
                    x=x,
                    y=y,
                    velocity_x=reader.f32(obj + FIGHTER_VELOCITY_X),
                    velocity_y=reader.f32(obj + FIGHTER_VELOCITY_Y),
                    acceleration_x=reader.f32(obj + 0xFC),
                    acceleration_y=reader.f32(obj + 0x100),
                    facing=reader.i8(obj + FIGHTER_FACING),
                    action_id=action_id,
                    frame_data=frame_data,
                    attack_flags=(
                        reader.u32(frame_data + FRAME_DATA_ATTACK_FLAGS)
                        if frame_data >= 0x10000 else 0
                    ),
                    body_box=_read_frame_box(reader, frame_data, FRAME_DATA_BODY_BOX),
                    hurt_boxes=read_frame_box_vector(
                        reader, frame_data, FRAME_DATA_BOX_VECTOR_A
                    ),
                    attack_boxes=read_frame_box_vector(
                        reader, frame_data, FRAME_DATA_BOX_VECTOR_B
                    ),
                )
            )
        node = reader.u32(node)
    if count and node != sentinel:
        raise RuntimeError("object-manager active list did not close at its sentinel")
    return tuple(projectiles)


def run_bootstrap_fight(
    reader: ProcessReader,
    keyboard: BattleKeyboard,
    *,
    seconds: float,
    frame_hz: float = 60.0,
) -> dict[str, int | float | str]:
    if seconds <= 0:
        raise ValueError("battle duration must be positive")
    if scene_id(reader) != SCENE_BATTLE:
        raise RuntimeError(f"fight requested outside battle scene: {scene_id(reader)}")

    deadline = time.perf_counter() + seconds
    cycles = 0
    reason = "duration"
    while time.perf_counter() < deadline:
        if scene_id(reader) != SCENE_BATTLE:
            reason = f"scene-{scene_id(reader)}"
            break
        play_combo(keyboard, BOOTSTRAP_CYCLE, frame_hz=frame_hz)
        cycles += 1
    keyboard.set_chord(set())
    return {"cycles": cycles, "seconds": seconds, "stop_reason": reason}


def run_adaptive_fight(
    reader: ProcessReader,
    keyboard: BattleKeyboard,
    *,
    seconds: float,
    frame_hz: float = 60.0,
    policy_path: Path | None = None,
    telemetry_path: Path | None = None,
    difficulty: str = "unknown",
    terminal_tracker: TerminalRoundTracker | None = None,
) -> dict[str, int | float | str | object]:
    """Sense/actuate shell around a fail-safe hot-reloadable combat policy."""
    if seconds <= 0 or frame_hz <= 0:
        raise ValueError("battle duration and frame rate must be positive")
    state = read_battle_state(reader)
    terminal_tracker = terminal_tracker or TerminalRoundTracker()
    initial = battle_state_json(state)
    deadline = time.perf_counter() + seconds
    period = 1.0 / frame_hz
    next_tick = time.perf_counter()
    frames = guard_frames = projectile_guard_frames = decision_frames = 0
    round_end_frames = 0
    round_wins = round_losses = round_draws = 0
    projectile_read_failures = consecutive_projectile_read_failures = 0
    max_consecutive_projectile_read_failures = 0
    terminal_reported = False
    last_enemy_projectiles: tuple[ProjectileState, ...] = ()
    last_own_projectiles: tuple[ProjectileState, ...] = ()
    nearest_enemy_projectile: float | None = None
    previous_state: BattleState | None = None
    last_hp_pair = (state.p1.hp, state.p2.hp)
    last_intent = "initializing"
    plugin = HotReloadPolicy(
        policy_path or Path(__file__).with_name("policies") / "adaptive.py"
    )
    reason = "duration"
    knowledge_path = (
        telemetry_path.with_name("th105_opponent_models.json")
        if telemetry_path is not None else None
    )
    base_opponent_key = f"0x{state.p2.vtable:08X}"
    opponent_key = f"{base_opponent_key}@{difficulty.casefold()}"
    # Old checkpoints were difficulty-agnostic. They remain useful priors, but
    # all new observations are stored under a difficulty-specific key so the
    # CPU's distinct timing distributions do not get averaged together.
    def contextual_prior(loader):
        return loader(knowledge_path, opponent_key) or loader(
            knowledge_path, base_opponent_key
        )

    prior_opponent_model = (
        contextual_prior(profiles_for)
        if knowledge_path is not None else {}
    )
    prior_projectile_model = (
        contextual_prior(projectile_model_for)
        if knowledge_path is not None else {}
    )
    prior_defense_model = (
        contextual_prior(defense_model_for)
        if knowledge_path is not None else {}
    )
    if knowledge_path is not None:
        prior_offense_model = offense_model_for(knowledge_path, opponent_key)
        if not prior_offense_model:
            prior_offense_model = cold_start_offense_prior(
                offense_model_for(knowledge_path, base_opponent_key)
            )
    else:
        prior_offense_model = {}
    prior_attack_geometry = (
        contextual_prior(attack_geometry_for)
        if knowledge_path is not None else {}
    )
    prior_cancel_graph = (
        contextual_prior(cancel_graph_for)
        if knowledge_path is not None else {}
    )
    prior_human_demonstrations: dict[str, object] = {}
    if telemetry_path is not None:
        human_source = telemetry_path.with_name("th105_human_demonstrations.json")
        human_compiled = telemetry_path.with_name("th105_human_policy.json")
        if human_source.is_file() and (
            not human_compiled.is_file()
            or human_compiled.stat().st_mtime_ns < human_source.stat().st_mtime_ns
        ):
            compile_human_file(human_source, human_compiled)
        prior_human_demonstrations = compiled_demonstrations_for(
            human_compiled,
            f"0x{state.p1.vtable:08X}",
            base_opponent_key,
        )

    telemetry = BoundedJsonlWriter(telemetry_path) if telemetry_path is not None else None

    def compact_plugin_status(status: dict[str, object]) -> dict[str, object]:
        metrics = status.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        return {
            "name": status.get("name"),
            "generation": status.get("generation"),
            "reloads": status.get("reloads"),
            "reload_failures": status.get("reload_failures"),
            "last_error": status.get("last_error"),
            "sha256": status.get("sha256"),
            "metrics": {
                "last_risk": metrics.get("last_risk"),
                "last_damage_context": metrics.get("last_damage_context"),
                "opponent_assessment": metrics.get("opponent_assessment"),
                "projectile_envelopes": metrics.get("projectile_envelopes"),
                "human_demonstrations": metrics.get("human_demonstrations"),
                "attack_geometry": metrics.get("attack_geometry"),
                "cancel_graph": metrics.get("cancel_graph"),
                "performance": metrics.get("performance"),
            },
        }

    def emit(event: str, payload: dict[str, object]) -> None:
        if telemetry is None:
            return
        record = {"time": time.time(), "event": event, **payload}
        telemetry.write(record)

    def persist_current_models(*, compile_models: bool) -> bool:
        """Atomically checkpoint bounded aggregates during long encounters."""
        # Scene 5 can briefly reappear with dead fighters during transitions.
        # Such a shell never seeded its plugin and must not erase cumulative
        # knowledge with an empty state.
        if knowledge_path is None or decision_frames == 0:
            return False
        status = plugin.status()
        metrics = status.get("metrics", {})
        if not isinstance(metrics, dict):
            return False
        learned_profiles = metrics.get("opponent_model", {})
        if not isinstance(learned_profiles, dict):
            return False
        learned_projectiles = metrics.get("projectile_envelope_state", {})
        learned_defenses = metrics.get("defense_response_state", {})
        learned_offense = metrics.get("offense_outcome_state", {})
        learned_attack_geometry = metrics.get("attack_geometry_state", {})
        learned_cancel_graph = metrics.get("cancel_graph_state", {})
        persist_character_models(
            knowledge_path,
            opponent_key,
            profiles=learned_profiles,
            projectile_envelopes=(
                learned_projectiles if isinstance(learned_projectiles, dict) else {}
            ),
            defense_responses=(
                learned_defenses if isinstance(learned_defenses, dict) else {}
            ),
            offense_outcomes=(
                learned_offense if isinstance(learned_offense, dict) else {}
            ),
            attack_geometry=(
                learned_attack_geometry
                if isinstance(learned_attack_geometry, dict) else {}
            ),
            cancel_graph=(
                learned_cancel_graph if isinstance(learned_cancel_graph, dict) else {}
            ),
        )
        if compile_models:
            compile_knowledge_file(
                knowledge_path,
                knowledge_path.with_name("th105_compiled_policy.json"),
            )
        return True

    emit(
        "encounter-start",
        {
            "initial": initial,
            "plugin": plugin.status(),
            "frame_hz": frame_hz,
            "difficulty": difficulty,
            "opponent": opponent_key,
        },
    )

    while time.perf_counter() < deadline:
        if scene_id(reader) != SCENE_BATTLE:
            reason = f"scene-{scene_id(reader)}"
            break
        try:
            state = read_battle_state(reader)
        except (OSError, RuntimeError):
            # Scene transitions can invalidate the fighter manager one frame
            # before the scene byte changes. End normally so learned state is
            # checkpointed instead of escaping to the outer recovery loop.
            reason = "battle-state-transition"
            break
        me, enemy = state.p1, state.p2
        if me.hp <= 0 or enemy.hp <= 0:
            if not terminal_reported:
                won = (
                    True if enemy.hp <= 0 < me.hp
                    else False if me.hp <= 0 < enemy.hp
                    else None
                )
                terminal_sequence = terminal_tracker.observe(
                    me_hp=me.hp, enemy_hp=enemy.hp
                )
                if terminal_sequence is not None:
                    plugin.observe_terminal(
                        {
                            "frame": frames,
                            "won": won,
                            "me_hp": me.hp,
                            "enemy_hp": enemy.hp,
                            "max_me_hp": me.max_hp,
                            "max_enemy_hp": enemy.max_hp,
                            "spirit": me.spirit,
                        }
                    )
                    if won is True:
                        round_wins += 1
                    elif won is False:
                        round_losses += 1
                    else:
                        round_draws += 1
                    emit(
                        "round-terminal",
                        {
                            "terminal_sequence": terminal_sequence,
                            "manager": f"0x{state.manager:08X}",
                            "opponent": opponent_key,
                            "difficulty": difficulty,
                            "won": won,
                            "me_hp": me.hp,
                            "enemy_hp": enemy.hp,
                            "max_me_hp": me.max_hp,
                            "max_enemy_hp": enemy.max_hp,
                        },
                    )
                terminal_reported = True
            # Z rising edges advance round-result text and post-fight dialogue.
            # Keep them sparse so one edge cannot leak through several screens.
            keys = {"z"} if round_end_frames % 30 == 0 else set()
            keyboard.set_chord(keys)
            last_intent = "round-end-confirm"
            last_enemy_projectiles = ()
            last_own_projectiles = ()
            nearest_enemy_projectile = None
            round_end_frames += 1
        else:
            terminal_tracker.observe(me_hp=me.hp, enemy_hp=enemy.hp)
            round_end_frames = 0
            terminal_reported = False
            try:
                enemy_projectiles = read_active_projectiles(reader, enemy, me)
                own_projectiles = read_active_projectiles(reader, me, enemy)
                last_enemy_projectiles = enemy_projectiles
                last_own_projectiles = own_projectiles
                consecutive_projectile_read_failures = 0
            except (OSError, RuntimeError) as exc:
                # The intrusive list can mutate between its begin/end reads as
                # projectiles spawn or expire. Fighter state remains valid, and
                # ProjectileState contains copied values, so reuse at most two
                # snapshots instead of destroying the policy's active episode.
                projectile_read_failures += 1
                consecutive_projectile_read_failures += 1
                max_consecutive_projectile_read_failures = max(
                    max_consecutive_projectile_read_failures,
                    consecutive_projectile_read_failures,
                )
                if consecutive_projectile_read_failures > 2:
                    last_enemy_projectiles = ()
                    last_own_projectiles = ()
                if consecutive_projectile_read_failures == 1:
                    emit(
                        "projectile-read-race",
                        {"frame": frames, "error": str(exc)[:160]},
                    )
                if consecutive_projectile_read_failures >= 60:
                    reason = "projectile-state-unavailable"
                    break
            nearest_enemy_projectile = None
            for projectile in last_enemy_projectiles:
                distance_to_projectile = math.hypot(
                    projectile.x - me.x, projectile.y - me.y
                )
                if nearest_enemy_projectile is None or distance_to_projectile < nearest_enemy_projectile:
                    nearest_enemy_projectile = distance_to_projectile
            decision = plugin.decide(
                PolicyObservation(
                    frame=frames,
                    state=state,
                    previous_state=previous_state,
                    enemy_projectiles=last_enemy_projectiles,
                    own_projectiles=last_own_projectiles,
                    prior_opponent_model=prior_opponent_model,
                    prior_projectile_model=prior_projectile_model,
                    prior_defense_model=prior_defense_model,
                    prior_offense_model=prior_offense_model,
                    prior_human_demonstrations=prior_human_demonstrations,
                    prior_attack_geometry=prior_attack_geometry,
                    prior_cancel_graph=prior_cancel_graph,
                )
            )
            decision_frames += 1
            last_intent = decision.intent
            if "guard" in decision.intent:
                guard_frames += 1
            if decision.intent == "projectile-guard":
                projectile_guard_frames += 1
            keyboard.set_chord(set(decision.keys))

        hp_pair = (me.hp, enemy.hp)
        if hp_pair != last_hp_pair:
            emit(
                "hp-change",
                {
                    "frame": frames,
                    "intent": last_intent,
                    "previous_hp": {"p1": last_hp_pair[0], "p2": last_hp_pair[1]},
                    "state": battle_state_json(state),
                    "enemy_projectiles": len(last_enemy_projectiles),
                    "own_projectiles": len(last_own_projectiles),
                    "plugin": compact_plugin_status(plugin.status()),
                },
            )
            last_hp_pair = hp_pair

        previous_state = state
        if frames and frames % 60 == 0:
            emit(
                "heartbeat",
                {
                    "frame": frames,
                    "intent": last_intent,
                    "state": battle_state_json(state),
                    "enemy_projectiles": len(last_enemy_projectiles),
                    "own_projectiles": len(last_own_projectiles),
                    "nearest_enemy_projectile": nearest_enemy_projectile,
                    "plugin": compact_plugin_status(plugin.status()),
                },
            )
        if frames and frames % max(1, int(frame_hz * 10.0)) == 0:
            if persist_current_models(compile_models=False):
                emit(
                    "model-checkpoint",
                    {"frame": frames, "opponent": opponent_key},
                )

        frames += 1
        next_tick += period
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.perf_counter()

    keyboard.set_chord(set())
    final = None
    if scene_id(reader) == SCENE_BATTLE:
        try:
            final = battle_state_json(read_battle_state(reader))
        except (OSError, RuntimeError):
            if reason == "duration":
                reason = "battle-state-transition"
    result = {
        "policy": "hot-reload",
        "frames": frames,
        "decision_frames": decision_frames,
        "guard_frames": guard_frames,
        "projectile_guard_frames": projectile_guard_frames,
        "enemy_projectiles": len(last_enemy_projectiles),
        "own_projectiles": len(last_own_projectiles),
        "nearest_enemy_projectile": nearest_enemy_projectile,
        "last_intent": last_intent,
        "round_end_frames": round_end_frames,
        "round_wins": round_wins,
        "round_losses": round_losses,
        "round_draws": round_draws,
        "projectile_read_failures": projectile_read_failures,
        "max_consecutive_projectile_read_failures": (
            max_consecutive_projectile_read_failures
        ),
        "difficulty": difficulty,
        "opponent": opponent_key,
        "seconds": seconds,
        "stop_reason": reason,
        "initial": initial,
        "final": final,
        "plugin": plugin.status(),
    }
    persist_current_models(compile_models=True)
    emit("encounter-end", result)
    return result
