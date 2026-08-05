"""Reference/native projectile path risk evaluation with a stable ABI."""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HazardProjectile:
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    half_width: float = 16.0
    half_height: float = 16.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0


@dataclass(frozen=True)
class MovementCandidate:
    velocity_x: float
    velocity_y: float
    half_width: float = 0.0
    half_height: float = 0.0
    graze_frames: int = 0
    startup_frames: int = 0


@dataclass(frozen=True)
class RiskResult:
    safe: bool
    first_collision_frame: int
    minimum_clearance: float
    final_x: float
    final_y: float


def _signed_clearance(
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
    projectile_x: float,
    projectile_y: float,
    projectile_half_width: float,
    projectile_half_height: float,
) -> float:
    gap_x = abs(player_x - projectile_x) - player_half_width - projectile_half_width
    gap_y = abs(player_y - projectile_y) - player_half_height - projectile_half_height
    if gap_x <= 0.0 and gap_y <= 0.0:
        return max(gap_x, gap_y)
    return math.hypot(max(0.0, gap_x), max(0.0, gap_y))


def evaluate_paths_reference(
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
    projectiles: tuple[HazardProjectile, ...],
    candidates: tuple[MovementCandidate, ...],
    *,
    horizon: int = 12,
    collision_margin: float = 0.0,
) -> tuple[RiskResult, ...]:
    if not 0 < horizon <= 600 or not candidates:
        raise ValueError("invalid horizon or empty candidates")
    results: list[RiskResult] = []
    for candidate in candidates:
        if not 0 <= candidate.startup_frames <= horizon:
            raise ValueError("candidate startup outside horizon")
        half_width = candidate.half_width or player_half_width
        half_height = candidate.half_height or player_half_height
        minimum = math.inf
        first_collision = -1
        for frame in range(1, horizon + 1):
            movement_frame = max(0, frame - candidate.startup_frames)
            x = player_x + candidate.velocity_x * movement_frame
            y = player_y + candidate.velocity_y * movement_frame
            if (
                candidate.startup_frames < frame
                <= candidate.startup_frames + candidate.graze_frames
            ):
                continue
            for projectile in projectiles:
                clearance = _signed_clearance(
                    x,
                    y,
                    half_width,
                    half_height,
                    projectile.x + projectile.velocity_x * frame
                    + 0.5 * projectile.acceleration_x * frame * frame,
                    projectile.y + projectile.velocity_y * frame
                    + 0.5 * projectile.acceleration_y * frame * frame,
                    projectile.half_width,
                    projectile.half_height,
                )
                minimum = min(minimum, clearance)
                if clearance <= collision_margin and first_collision < 0:
                    first_collision = frame
        results.append(
            RiskResult(
                safe=first_collision < 0,
                first_collision_frame=first_collision,
                minimum_clearance=minimum,
                final_x=player_x + candidate.velocity_x * max(0, horizon - candidate.startup_frames),
                final_y=player_y + candidate.velocity_y * max(0, horizon - candidate.startup_frames),
            )
        )
    return tuple(results)


class _Projectile(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("half_width", ctypes.c_float),
        ("half_height", ctypes.c_float),
        ("acceleration_x", ctypes.c_float),
        ("acceleration_y", ctypes.c_float),
    )


class _Candidate(ctypes.Structure):
    _fields_ = (
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("half_width", ctypes.c_float),
        ("half_height", ctypes.c_float),
        ("graze_frames", ctypes.c_int32),
        ("startup_frames", ctypes.c_int32),
    )


class _Result(ctypes.Structure):
    _fields_ = (
        ("safe", ctypes.c_int32),
        ("first_collision_frame", ctypes.c_int32),
        ("minimum_clearance", ctypes.c_float),
        ("final_x", ctypes.c_float),
        ("final_y", ctypes.c_float),
    )


class NativeHazardKernel:
    ABI_VERSION = 4

    def __init__(self, path: Path | None = None) -> None:
        if os.name != "nt":
            raise RuntimeError("native TH105 hazard kernel requires Windows Python")
        dll_path = path or Path(__file__).resolve().parents[2] / "build" / "th105_hazard.dll"
        if not dll_path.is_file():
            raise RuntimeError("missing build/th105_hazard.dll; run ./build_th105_native.sh")
        self.library = ctypes.CDLL(str(dll_path))
        version = self.library.th105_hazard_abi_version
        version.argtypes = ()
        version.restype = ctypes.c_int32
        if version() != self.ABI_VERSION:
            raise RuntimeError("TH105 native hazard ABI mismatch")
        self.function = self.library.th105_evaluate_linear_paths
        self.function.argtypes = (
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.POINTER(_Projectile),
            ctypes.c_uint32,
            ctypes.POINTER(_Candidate),
            ctypes.c_uint32,
            ctypes.POINTER(_Result),
        )
        self.function.restype = ctypes.c_int32

    def evaluate(
        self,
        player_x: float,
        player_y: float,
        player_half_width: float,
        player_half_height: float,
        projectiles: tuple[HazardProjectile, ...],
        candidates: tuple[MovementCandidate, ...],
        *,
        horizon: int = 12,
        collision_margin: float = 0.0,
    ) -> tuple[RiskResult, ...]:
        projectile_array = (_Projectile * len(projectiles))(
            *(_Projectile(*vars(item).values()) for item in projectiles)
        )
        candidate_array = (_Candidate * len(candidates))(
            *(
                _Candidate(
                    item.velocity_x,
                    item.velocity_y,
                    item.half_width,
                    item.half_height,
                    item.graze_frames,
                    item.startup_frames,
                )
                for item in candidates
            )
        )
        output = (_Result * len(candidates))()
        count = self.function(
            player_x,
            player_y,
            player_half_width,
            player_half_height,
            collision_margin,
            horizon,
            projectile_array,
            len(projectiles),
            candidate_array,
            len(candidates),
            output,
        )
        if count != len(candidates):
            raise RuntimeError(f"native hazard evaluation failed: {count}")
        return tuple(
            RiskResult(
                bool(item.safe),
                item.first_collision_frame,
                item.minimum_clearance,
                item.final_x,
                item.final_y,
            )
            for item in output
        )


class HazardEvaluator:
    """Use the DLL on Windows when present, otherwise retain reference parity."""

    def __init__(self) -> None:
        try:
            self.native: NativeHazardKernel | None = NativeHazardKernel()
        except RuntimeError:
            self.native = None

    @property
    def backend(self) -> str:
        return "native" if self.native is not None else "python"

    def evaluate(self, *args, **kwargs) -> tuple[RiskResult, ...]:
        if self.native is not None:
            return self.native.evaluate(*args, **kwargs)
        return evaluate_paths_reference(*args, **kwargs)
