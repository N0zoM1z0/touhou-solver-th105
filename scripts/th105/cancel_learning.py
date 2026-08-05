"""Bounded, input-grounded cancel graph learned from actual fighter transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ALLOWED_KEYS = frozenset({"toward", "back", "up", "down", "a", "z", "x", "c"})


@dataclass
class CancelEdgeStats:
    source_action: int
    source_sequence: int
    source_pose: int
    target_action: int
    target_sequence: int
    chord: str
    trials: int = 0
    direct_damage_events: int = 0
    total_direct_damage: int = 0
    minimum_source_frame: int = 0
    maximum_source_frame: int = 0


class CancelGraphModel:
    """Records only transitions the native game actually accepted."""

    MAX_EDGES = 2048

    def __init__(self) -> None:
        self.edges: dict[str, CancelEdgeStats] = {}
        self.pending_input: dict[str, object] | None = None
        self.last_attack_buttons: frozenset[str] = frozenset()
        self.input_edges = 0
        self.accepted_transitions = 0
        self.expired_inputs = 0

    @staticmethod
    def _offensive(action: int) -> bool:
        return 300 <= action < 600

    @staticmethod
    def edge_id(
        source_action: int,
        source_sequence: int,
        source_pose: int,
        chord: str,
        target_action: int,
        target_sequence: int,
    ) -> str:
        return (
            f"{source_action}:{source_sequence}:{source_pose}|{chord}|"
            f"{target_action}:{target_sequence}"
        )

    def observe_transition(
        self,
        *,
        source_action: int,
        source_sequence: int,
        source_pose: int,
        source_frame: int,
        target_action: int,
        target_sequence: int,
        normalized_keys: frozenset[str],
        direct_damage: int = 0,
    ) -> None:
        if (
            not self._offensive(source_action)
            or not self._offensive(target_action)
            or (source_action, source_sequence) == (target_action, target_sequence)
            or not normalized_keys & {"z", "x", "c"}
            or not normalized_keys <= ALLOWED_KEYS
        ):
            return
        chord = "+".join(sorted(normalized_keys))
        edge_id = self.edge_id(
            source_action,
            source_sequence,
            source_pose,
            chord,
            target_action,
            target_sequence,
        )
        row = self.edges.get(edge_id)
        if row is None:
            if len(self.edges) >= self.MAX_EDGES:
                return
            row = CancelEdgeStats(
                source_action,
                source_sequence,
                source_pose,
                target_action,
                target_sequence,
                chord,
                minimum_source_frame=max(0, int(source_frame)),
                maximum_source_frame=max(0, int(source_frame)),
            )
            self.edges[edge_id] = row
        row.trials += 1
        row.minimum_source_frame = min(row.minimum_source_frame, max(0, int(source_frame)))
        row.maximum_source_frame = max(row.maximum_source_frame, max(0, int(source_frame)))
        damage = max(0, int(direct_damage))
        if damage:
            row.direct_damage_events += 1
            row.total_direct_damage += damage

    def record_input(
        self,
        *,
        frame: int,
        source_action: int,
        source_sequence: int,
        source_pose: int,
        source_frame: int,
        normalized_keys: frozenset[str],
    ) -> None:
        attack_buttons = normalized_keys & {"z", "x", "c"}
        rising = attack_buttons - self.last_attack_buttons
        self.last_attack_buttons = attack_buttons
        if not rising or not self._offensive(source_action):
            return
        self.pending_input = {
            "issued_frame": int(frame),
            "source_action": int(source_action),
            "source_sequence": int(source_sequence),
            "source_pose": int(source_pose),
            "source_frame": int(source_frame),
            "keys": tuple(sorted(normalized_keys)),
        }
        self.input_edges += 1

    def observe_actor(
        self,
        *,
        frame: int,
        action_id: int,
        sequence: int,
        direct_damage: int = 0,
        maximum_cause_age: int = 6,
    ) -> bool:
        pending = self.pending_input
        if pending is None:
            return False
        source_signature = (
            int(pending["source_action"]),
            int(pending["source_sequence"]),
        )
        age = int(frame) - int(pending["issued_frame"])
        if (int(action_id), int(sequence)) != source_signature:
            before = len(self.edges)
            prior_trials = sum(row.trials for row in self.edges.values())
            self.observe_transition(
                source_action=source_signature[0],
                source_sequence=source_signature[1],
                source_pose=int(pending["source_pose"]),
                source_frame=int(pending["source_frame"]),
                target_action=int(action_id),
                target_sequence=int(sequence),
                normalized_keys=frozenset(pending["keys"]),
                direct_damage=direct_damage,
            )
            self.pending_input = None
            accepted = len(self.edges) > before or sum(
                row.trials for row in self.edges.values()
            ) > prior_trials
            if accepted:
                self.accepted_transitions += 1
            return accepted
        if age > maximum_cause_age or not self._offensive(action_id):
            self.pending_input = None
            self.expired_inputs += 1
        return False

    def reset_episode(self) -> None:
        self.pending_input = None
        self.last_attack_buttons = frozenset()

    def candidates(
        self,
        *,
        action_id: int,
        sequence: int,
        pose: int,
        action_frame: int,
        timing_margin: int = 2,
    ) -> tuple[tuple[str, CancelEdgeStats], ...]:
        viable = tuple(
            (edge_id, row)
            for edge_id, row in self.edges.items()
            if row.source_action == action_id
            and row.source_sequence == sequence
            and row.source_pose == pose
            and row.minimum_source_frame - timing_margin
            <= action_frame
            <= row.maximum_source_frame + timing_margin
        )
        return tuple(
            sorted(
                viable,
                key=lambda item: (
                    item[1].direct_damage_events,
                    item[1].trials,
                    item[0],
                ),
                reverse=True,
            )
        )

    def export_state(self) -> dict[str, object]:
        return {edge_id: asdict(row) for edge_id, row in self.edges.items()}

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        for edge_id, raw in list(state.items())[: self.MAX_EDGES]:
            if not isinstance(raw, dict):
                continue
            chord = str(raw.get("chord", ""))
            keys = frozenset(chord.split("+")) if chord else frozenset()
            if not keys or not keys <= ALLOWED_KEYS:
                continue
            self.edges[str(edge_id)] = CancelEdgeStats(
                source_action=int(raw.get("source_action", 0)),
                source_sequence=int(raw.get("source_sequence", 0)),
                source_pose=int(raw.get("source_pose", 0)),
                target_action=int(raw.get("target_action", 0)),
                target_sequence=int(raw.get("target_sequence", 0)),
                chord=chord,
                trials=max(0, int(raw.get("trials", 0))),
                direct_damage_events=max(0, int(raw.get("direct_damage_events", 0))),
                total_direct_damage=max(0, int(raw.get("total_direct_damage", 0))),
                minimum_source_frame=max(0, int(raw.get("minimum_source_frame", 0))),
                maximum_source_frame=max(0, int(raw.get("maximum_source_frame", 0))),
            )

    def metrics(self) -> dict[str, int]:
        return {
            "edges": len(self.edges),
            "trials": sum(row.trials for row in self.edges.values()),
            "direct_damage_events": sum(
                row.direct_damage_events for row in self.edges.values()
            ),
            "input_edges": self.input_edges,
            "accepted_transitions": self.accepted_transitions,
            "expired_inputs": self.expired_inputs,
            "pending": int(self.pending_input is not None),
        }
