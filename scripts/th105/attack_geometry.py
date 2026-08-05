"""Bounded native attack-box timing learned per action/sequence/pose."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


Box = tuple[int, int, int, int]


def _box_envelope(boxes: tuple[Box, ...]) -> Box | None:
    if not boxes:
        return None
    sane = tuple(
        box
        for box in boxes[:32]
        if len(box) == 4
        and box[0] <= box[2]
        and box[1] <= box[3]
        and all(abs(int(value)) <= 4096 for value in box)
    )
    if not sane:
        return None
    return (
        min(box[0] for box in sane),
        min(box[1] for box in sane),
        max(box[2] for box in sane),
        max(box[3] for box in sane),
    )


def _merge_envelope(first: Box | None, second: Box) -> Box:
    if first is None:
        return second
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


@dataclass
class PoseGeometry:
    pose: int
    observations: int = 0
    active_observations: int = 0
    first_elapsed: int = 0
    last_elapsed: int = 0
    attack_envelope: Box | None = None


@dataclass
class ActionGeometry:
    action_id: int
    sequence: int
    observations: int = 0
    active_observations: int = 0
    first_active_elapsed: int = 0
    last_active_elapsed: int = 0
    attack_envelope: Box | None = None
    poses: dict[int, PoseGeometry] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackForecast:
    action_id: int
    sequence: int
    first_active_elapsed: int
    last_active_elapsed: int
    attack_envelope: Box
    active_observations: int


class AttackGeometryModel:
    """Constant-cardinality sufficient statistics over native local AABBs."""

    MAX_ACTIONS = 256
    MAX_POSES_PER_ACTION = 64

    def __init__(self) -> None:
        self.actions: dict[tuple[int, int], ActionGeometry] = {}
        self.current_signature: tuple[int, int] | None = None
        self.start_frame = 0

    @staticmethod
    def is_modelled(action_id: int) -> bool:
        return 300 <= action_id < 800

    def reset_episode(self) -> None:
        self.current_signature = None
        self.start_frame = 0

    def observe(
        self,
        frame: int,
        *,
        action_id: int,
        sequence: int,
        pose: int,
        attack_boxes: tuple[Box, ...],
    ) -> None:
        signature = (int(action_id), int(sequence))
        if not self.is_modelled(action_id):
            self.reset_episode()
            return
        if signature != self.current_signature:
            self.current_signature = signature
            self.start_frame = int(frame)
        elapsed = max(0, int(frame) - self.start_frame)
        action = self.actions.get(signature)
        if action is None:
            if len(self.actions) >= self.MAX_ACTIONS:
                return
            action = ActionGeometry(*signature)
            self.actions[signature] = action
        action.observations += 1
        pose_id = int(pose)
        pose_row = action.poses.get(pose_id)
        if pose_row is None:
            if len(action.poses) >= self.MAX_POSES_PER_ACTION:
                return
            pose_row = PoseGeometry(pose_id, first_elapsed=elapsed, last_elapsed=elapsed)
            action.poses[pose_id] = pose_row
        pose_row.observations += 1
        pose_row.first_elapsed = min(pose_row.first_elapsed, elapsed)
        pose_row.last_elapsed = max(pose_row.last_elapsed, elapsed)
        envelope = _box_envelope(attack_boxes)
        if envelope is None:
            return
        pose_row.active_observations += 1
        pose_row.attack_envelope = _merge_envelope(pose_row.attack_envelope, envelope)
        if action.active_observations == 0:
            action.first_active_elapsed = elapsed
        else:
            action.first_active_elapsed = min(action.first_active_elapsed, elapsed)
        action.active_observations += 1
        action.last_active_elapsed = max(action.last_active_elapsed, elapsed)
        action.attack_envelope = _merge_envelope(action.attack_envelope, envelope)

    def current_elapsed(self, frame: int, action_id: int, sequence: int) -> int | None:
        if self.current_signature != (int(action_id), int(sequence)):
            return None
        return max(0, int(frame) - self.start_frame)

    def forecast(self, action_id: int, sequence: int) -> AttackForecast | None:
        row = self.actions.get((int(action_id), int(sequence)))
        if row is None or row.attack_envelope is None or not row.active_observations:
            return None
        return AttackForecast(
            row.action_id,
            row.sequence,
            row.first_active_elapsed,
            row.last_active_elapsed,
            row.attack_envelope,
            row.active_observations,
        )

    def export_state(self) -> dict[str, object]:
        return {
            f"{action_id}:{sequence}": {
                "action_id": row.action_id,
                "sequence": row.sequence,
                "observations": row.observations,
                "active_observations": row.active_observations,
                "first_active_elapsed": row.first_active_elapsed,
                "last_active_elapsed": row.last_active_elapsed,
                "attack_envelope": list(row.attack_envelope) if row.attack_envelope else None,
                "poses": {
                    str(pose): {
                        **asdict(pose_row),
                        "attack_envelope": (
                            list(pose_row.attack_envelope)
                            if pose_row.attack_envelope else None
                        ),
                    }
                    for pose, pose_row in row.poses.items()
                },
            }
            for (action_id, sequence), row in self.actions.items()
        }

    @staticmethod
    def _decode_envelope(raw: object) -> Box | None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        box = tuple(int(value) for value in raw)
        assert len(box) == 4
        return _box_envelope((box,))

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        for _key, raw in list(state.items())[: self.MAX_ACTIONS]:
            if not isinstance(raw, dict):
                continue
            action_id = int(raw.get("action_id", -1))
            sequence = int(raw.get("sequence", -1))
            if not self.is_modelled(action_id) or not 0 <= sequence <= 0xFFFF:
                continue
            row = ActionGeometry(
                action_id,
                sequence,
                observations=max(0, int(raw.get("observations", 0))),
                active_observations=max(0, int(raw.get("active_observations", 0))),
                first_active_elapsed=max(0, int(raw.get("first_active_elapsed", 0))),
                last_active_elapsed=max(0, int(raw.get("last_active_elapsed", 0))),
                attack_envelope=self._decode_envelope(raw.get("attack_envelope")),
            )
            raw_poses = raw.get("poses", {})
            if isinstance(raw_poses, dict):
                for raw_pose, raw_pose_row in list(raw_poses.items())[
                    : self.MAX_POSES_PER_ACTION
                ]:
                    if not isinstance(raw_pose_row, dict):
                        continue
                    pose = int(raw_pose)
                    row.poses[pose] = PoseGeometry(
                        pose,
                        observations=max(0, int(raw_pose_row.get("observations", 0))),
                        active_observations=max(
                            0, int(raw_pose_row.get("active_observations", 0))
                        ),
                        first_elapsed=max(0, int(raw_pose_row.get("first_elapsed", 0))),
                        last_elapsed=max(0, int(raw_pose_row.get("last_elapsed", 0))),
                        attack_envelope=self._decode_envelope(
                            raw_pose_row.get("attack_envelope")
                        ),
                    )
            self.actions[(action_id, sequence)] = row

    def metrics(self) -> dict[str, int]:
        return {
            "actions": len(self.actions),
            "poses": sum(len(row.poses) for row in self.actions.values()),
            "active_actions": sum(bool(row.active_observations) for row in self.actions.values()),
            "active_observations": sum(row.active_observations for row in self.actions.values()),
        }
