"""Read-only human demonstration aggregation for bounded offense priors."""

from __future__ import annotations

import json
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .offense_learning import combat_context
from .telemetry import BoundedJsonlWriter

if TYPE_CHECKING:
    from .battle import BattleState


def normalize_human_pattern(pattern: str) -> str:
    """Canonicalize a bounded trace and neutralize opposite directions."""
    allowed = {"up", "down", "toward", "back", "z", "x", "c", "a"}
    runs: list[tuple[str, int]] = []
    total_frames = 0
    for item in pattern.split(","):
        chord_text, separator, frame_text = item.rpartition("@")
        if not separator:
            return ""
        try:
            frames = int(frame_text)
        except ValueError:
            return ""
        if frames <= 0 or frames > 180:
            return ""
        keys = set() if chord_text == "neutral" else set(chord_text.split("+"))
        if not keys <= allowed:
            return ""
        if {"toward", "back"} <= keys:
            keys -= {"toward", "back"}
        if {"up", "down"} <= keys:
            keys -= {"up", "down"}
        chord = "+".join(sorted(keys)) if keys else "neutral"
        total_frames += frames
        if total_frames > 180:
            return ""
        if runs and runs[-1][0] == chord:
            runs[-1] = (chord, runs[-1][1] + frames)
        else:
            runs.append((chord, frames))
    encoded: list[str] = []
    for chord, frames in runs:
        while frames > 60:
            encoded.append(f"{chord}@60")
            frames -= 60
        encoded.append(f"{chord}@{frames}")
    return ",".join(encoded)


def human_demonstration_utility(
    raw: dict[str, object],
    *,
    spirit: int,
    support: int,
    context_penalty: float = 0.0,
) -> float | None:
    """Return safe empirical utility, or None for an unusable demonstration."""
    trials = max(1, int(raw.get("trials", 0)))
    connections = int(raw.get("connections", 0))
    if connections <= 0 or connections / trials < 0.5:
        return None
    average_damage = float(raw.get("total_damage", 0)) / trials
    average_self_damage = float(raw.get("total_self_damage", 0)) / trials
    average_spirit = float(raw.get("total_spirit_cost", 0)) / trials
    average_duration = float(raw.get("total_duration", 0)) / trials
    if average_spirit > max(0, spirit - 200):
        return None
    if average_self_damage > max(100.0, average_damage * 0.35):
        return None
    score = (
        average_damage
        - 1.5 * average_self_damage
        - 0.2 * average_spirit
        - 1.5 * average_duration
        + min(5, support) * 40.0
        - context_penalty
    )
    return score if score > 0.0 else None


def compile_human_demonstrations(
    data: dict[str, object],
    *,
    assumed_spirit: int = 1000,
    max_candidates_per_context: int = 8,
) -> dict[str, object]:
    """Compile bounded raw aggregates into ranked online candidate lists."""
    output: dict[str, object] = {
        "version": 1,
        "generated_at": time.time(),
        "matchups": {},
    }
    matchups = data.get("matchups", {})
    if not isinstance(matchups, dict):
        return output
    compiled_matchups = output["matchups"]
    assert isinstance(compiled_matchups, dict)
    for matchup_key, raw_matchup in matchups.items():
        if not isinstance(raw_matchup, dict):
            continue
        contexts: dict[str, object] = {}
        raw_contexts = raw_matchup.get("chains", {})
        if isinstance(raw_contexts, dict):
            for context, raw_row in raw_contexts.items():
                if not isinstance(raw_row, dict):
                    continue
                candidates: list[dict[str, object]] = []
                for signature, raw in raw_row.items():
                    if signature == "__other__" or not isinstance(raw, dict):
                        continue
                    patterns = raw.get("input_patterns", {})
                    if not isinstance(patterns, dict) or not patterns:
                        continue
                    normalized_patterns: dict[str, dict[str, int]] = {}
                    for value, count in patterns.items():
                        normalized = normalize_human_pattern(str(value))
                        if normalized:
                            aggregate = normalized_patterns.setdefault(
                                normalized,
                                {
                                    "legacy_support": 0,
                                    "trials": 0,
                                    "connections": 0,
                                    "total_damage": 0,
                                    "total_self_damage": 0,
                                    "total_spirit_cost": 0,
                                    "total_duration": 0,
                                },
                            )
                            if isinstance(count, dict):
                                aggregate["legacy_support"] += int(
                                    count.get("legacy_support", 0)
                                )
                                for field in (
                                    "trials",
                                    "connections",
                                    "total_damage",
                                    "total_self_damage",
                                    "total_spirit_cost",
                                    "total_duration",
                                ):
                                    aggregate[field] += int(count.get(field, 0))
                            else:
                                aggregate["legacy_support"] += int(count)
                    if not normalized_patterns:
                        continue
                    signature_candidates: list[dict[str, object]] = []
                    for pattern, pattern_stats in normalized_patterns.items():
                        precise_trials = int(pattern_stats["trials"])
                        support = precise_trials + int(pattern_stats["legacy_support"])
                        outcome = pattern_stats if precise_trials else raw
                        score = human_demonstration_utility(
                            outcome,
                            spirit=assumed_spirit,
                            support=support,
                        )
                        if score is None:
                            continue
                        trials = max(1, int(outcome.get("trials", 0)))
                        signature_candidates.append(
                            {
                                "signature": str(signature),
                                "pattern": pattern,
                                "pattern_id": hashlib.sha256(
                                    pattern.encode("utf-8")
                                ).hexdigest()[:10],
                                "score": score,
                                "trials": trials,
                                "precise_trials": precise_trials,
                                "connections": int(outcome.get("connections", 0)),
                                "average_damage": (
                                    float(outcome.get("total_damage", 0)) / trials
                                ),
                                "average_self_damage": (
                                    float(outcome.get("total_self_damage", 0)) / trials
                                ),
                                "average_spirit": (
                                    float(outcome.get("total_spirit_cost", 0)) / trials
                                ),
                                "support": support,
                            }
                        )
                    candidates.extend(
                        sorted(
                            signature_candidates,
                            key=lambda item: (
                                float(item["score"]),
                                int(item["support"]),
                                str(item["pattern_id"]),
                            ),
                            reverse=True,
                        )[:3]
                    )
                if candidates:
                    contexts[str(context)] = sorted(
                        candidates,
                        key=lambda item: (
                            float(item["score"]),
                            int(item["support"]),
                            str(item["signature"]),
                        ),
                        reverse=True,
                    )[:max_candidates_per_context]
        if contexts:
            compiled_matchups[str(matchup_key)] = {
                "p1_vtable": raw_matchup.get("p1_vtable"),
                "p2_vtable": raw_matchup.get("p2_vtable"),
                "contexts": contexts,
            }
    return output


def compile_human_file(source: Path, destination: Path) -> dict[str, object]:
    data = json.loads(source.read_text(encoding="utf-8"))
    compiled = compile_human_demonstrations(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return compiled


def compiled_demonstrations_for(
    path: Path, p1_vtable: str, p2_vtable: str
) -> dict[str, object]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("matchups"), dict):
        return {}
    row = data["matchups"].get(f"{p1_vtable}|{p2_vtable}", {})
    return row if isinstance(row, dict) else {}


def demonstrations_for(
    path: Path, p1_vtable: str, p2_vtable: str
) -> dict[str, object]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("matchups"), dict):
        return {}
    row = data["matchups"].get(f"{p1_vtable}|{p2_vtable}", {})
    return row if isinstance(row, dict) else {}


def _phase_for_enemy_action(action_id: int) -> str:
    if 50 <= action_id < 200:
        return "reaction"
    if 300 <= action_id < 800:
        return "danger"
    return "neutral"


def _context(state: BattleState) -> str:
    me, enemy = state.p1, state.p2
    return combat_context(
        distance=abs(enemy.x - me.x),
        enemy_y=enemy.y,
        enemy_x=enemy.x,
        phase=_phase_for_enemy_action(enemy.action_id),
    )


@dataclass
class _Episode:
    signature: str
    context: str
    start_frame: int
    start_enemy_hp: int
    start_me_hp: int
    start_spirit: int
    command_mask: int
    enemy_hp: int
    me_hp: int
    spirit: int
    first_hit_frame: int | None = None
    input_trace: list[str] = field(default_factory=list)


@dataclass
class _Chain:
    context: str
    start_frame: int
    start_enemy_hp: int
    start_me_hp: int
    start_spirit: int
    signatures: list[str] = field(default_factory=list)
    input_trace: list[str] = field(default_factory=list)


class HumanDemonstrationRecorder:
    """Store action/transition/chain sufficient statistics, never raw frames."""

    def __init__(
        self,
        output_path: Path,
        *,
        telemetry_path: Path | None = None,
    ) -> None:
        self.output_path = output_path
        self.telemetry = (
            BoundedJsonlWriter(telemetry_path) if telemetry_path is not None else None
        )
        self.data = self._load()
        self.episode: _Episode | None = None
        self.chain: _Chain | None = None
        self.last_signature: str | None = None
        self.character_key: str | None = None
        self.opponent_key: str | None = None
        self.action_episodes = 0
        self.chains = 0
        self.input_history: deque[str] = deque(maxlen=16)

    def _load(self) -> dict[str, object]:
        if not self.output_path.is_file():
            return {"version": 1, "matchups": {}}
        data = json.loads(self.output_path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("matchups"), dict):
            raise ValueError(f"unsupported human demonstration format: {self.output_path}")
        return data

    def _matchup(self) -> dict[str, object]:
        if self.character_key is None or self.opponent_key is None:
            raise RuntimeError("demonstration matchup is not initialized")
        key = f"{self.character_key}|{self.opponent_key}"
        matchups = self.data["matchups"]
        assert isinstance(matchups, dict)
        row = matchups.setdefault(
            key,
            {
                "p1_vtable": self.character_key,
                "p2_vtable": self.opponent_key,
                "actions": {},
                "transitions": {},
                "chains": {},
            },
        )
        assert isinstance(row, dict)
        return row

    @staticmethod
    def _outcome(
        table: dict[str, object],
        key: str,
        *,
        damage: int,
        self_damage: int,
        spirit_cost: int,
        duration: int,
        startup: int | None,
        command_mask: int | None = None,
        input_pattern: str | None = None,
        max_rows: int = 256,
    ) -> None:
        if key not in table and len(table) >= max_rows:
            key = "__other__"
            input_pattern = None
        stats = table.setdefault(
            key,
            {
                "trials": 0,
                "connections": 0,
                "total_damage": 0,
                "total_self_damage": 0,
                "punished_trials": 0,
                "total_spirit_cost": 0,
                "total_duration": 0,
                "startup_samples": 0,
                "total_startup_to_hit": 0,
                "command_masks": {},
                "input_patterns": {},
            },
        )
        assert isinstance(stats, dict)
        stats["trials"] = int(stats.get("trials", 0)) + 1
        stats["connections"] = int(stats.get("connections", 0)) + int(damage > 0)
        stats["total_damage"] = int(stats.get("total_damage", 0)) + damage
        stats["total_self_damage"] = int(stats.get("total_self_damage", 0)) + self_damage
        stats["punished_trials"] = int(stats.get("punished_trials", 0)) + int(self_damage > 0)
        stats["total_spirit_cost"] = int(stats.get("total_spirit_cost", 0)) + spirit_cost
        stats["total_duration"] = int(stats.get("total_duration", 0)) + duration
        if startup is not None:
            stats["startup_samples"] = int(stats.get("startup_samples", 0)) + 1
            stats["total_startup_to_hit"] = int(stats.get("total_startup_to_hit", 0)) + startup
        if command_mask is not None:
            masks = stats.setdefault("command_masks", {})
            assert isinstance(masks, dict)
            mask = f"0x{command_mask:08X}"
            masks[mask] = int(masks.get(mask, 0)) + 1
        if input_pattern:
            patterns = stats.setdefault("input_patterns", {})
            assert isinstance(patterns, dict)
            if input_pattern in patterns or len(patterns) < 32:
                pattern_stats = patterns.get(input_pattern)
                if isinstance(pattern_stats, int):
                    pattern_stats = {"legacy_support": pattern_stats}
                elif not isinstance(pattern_stats, dict):
                    pattern_stats = {}
                pattern_stats["trials"] = int(pattern_stats.get("trials", 0)) + 1
                pattern_stats["connections"] = int(
                    pattern_stats.get("connections", 0)
                ) + int(damage > 0)
                pattern_stats["total_damage"] = int(
                    pattern_stats.get("total_damage", 0)
                ) + damage
                pattern_stats["total_self_damage"] = int(
                    pattern_stats.get("total_self_damage", 0)
                ) + self_damage
                pattern_stats["total_spirit_cost"] = int(
                    pattern_stats.get("total_spirit_cost", 0)
                ) + spirit_cost
                pattern_stats["total_duration"] = int(
                    pattern_stats.get("total_duration", 0)
                ) + duration
                patterns[input_pattern] = pattern_stats

    @staticmethod
    def _normalized_chord(keys: frozenset[str], facing: int) -> str:
        normalized = set(keys) - {"left", "right"}
        if not {"left", "right"} <= keys:
            if "left" in keys:
                normalized.add("toward" if facing < 0 else "back")
            if "right" in keys:
                normalized.add("toward" if facing > 0 else "back")
        if {"up", "down"} <= normalized:
            normalized -= {"up", "down"}
        return "+".join(sorted(normalized)) if normalized else "neutral"

    @staticmethod
    def _compress_trace(trace: list[str]) -> str:
        trace = list(trace)
        while trace and trace[0] == "neutral":
            trace.pop(0)
        while trace and trace[-1] == "neutral":
            trace.pop()
        runs: list[tuple[str, int]] = []
        for chord in trace:
            if runs and runs[-1][0] == chord:
                runs[-1] = (chord, runs[-1][1] + 1)
            else:
                runs.append((chord, 1))
        return ",".join(f"{chord}@{frames}" for chord, frames in runs)

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self.telemetry is not None:
            self.telemetry.write({"time": time.time(), "event": event, **payload})

    def _begin_action(self, frame: int, state: BattleState) -> None:
        me, enemy = state.p1, state.p2
        signature = f"{me.action_id}:{me.action_sequence}"
        context = _context(state)
        self.episode = _Episode(
            signature,
            context,
            frame,
            enemy.hp,
            me.hp,
            int(me.spirit),
            int(me.command_mask),
            enemy.hp,
            me.hp,
            int(me.spirit),
            input_trace=list(self.input_history),
        )
        if self.chain is None:
            self.chain = _Chain(
                context,
                frame,
                enemy.hp,
                me.hp,
                int(me.spirit),
                input_trace=list(self.input_history),
            )
        if len(self.chain.signatures) < 16:
            self.chain.signatures.append(signature)
        if self.last_signature is not None:
            transitions = self._matchup().setdefault("transitions", {})
            assert isinstance(transitions, dict)
            edge = f"{self.last_signature}>{signature}"
            if edge not in transitions and len(transitions) >= 1024:
                self.last_signature = signature
                return
            item = transitions.setdefault(edge, {"trials": 0})
            assert isinstance(item, dict)
            item["trials"] = int(item.get("trials", 0)) + 1
        self.last_signature = signature

    def _finish_action(self, frame: int) -> None:
        episode = self.episode
        self.episode = None
        if episode is None:
            return
        damage = max(0, episode.start_enemy_hp - episode.enemy_hp)
        self_damage = max(0, episode.start_me_hp - episode.me_hp)
        spirit_cost = max(0, episode.start_spirit - episode.spirit)
        duration = max(1, frame - episode.start_frame)
        startup = (
            episode.first_hit_frame - episode.start_frame
            if episode.first_hit_frame is not None else None
        )
        actions = self._matchup().setdefault("actions", {})
        assert isinstance(actions, dict)
        context_row = actions.setdefault(episode.context, {})
        assert isinstance(context_row, dict)
        self._outcome(
            context_row,
            episode.signature,
            damage=damage,
            self_damage=self_damage,
            spirit_cost=spirit_cost,
            duration=duration,
            startup=startup,
            command_mask=episode.command_mask,
            input_pattern=self._compress_trace(episode.input_trace),
        )
        self.action_episodes += 1
        self._emit(
            "human-action",
            {
                "signature": episode.signature,
                "context": episode.context,
                "damage": damage,
                "self_damage": self_damage,
                "spirit_cost": spirit_cost,
                "duration": duration,
                "startup": startup,
            },
        )

    def _finish_chain(self, frame: int, state: BattleState | None) -> None:
        chain = self.chain
        self.chain = None
        self.last_signature = None
        if chain is None or state is None or not chain.signatures:
            return
        me, enemy = state.p1, state.p2
        damage = max(0, chain.start_enemy_hp - enemy.hp)
        self_damage = max(0, chain.start_me_hp - me.hp)
        spirit_cost = max(0, chain.start_spirit - int(me.spirit))
        duration = max(1, frame - chain.start_frame)
        label = ">".join(chain.signatures)
        chains = self._matchup().setdefault("chains", {})
        assert isinstance(chains, dict)
        context_row = chains.setdefault(chain.context, {})
        assert isinstance(context_row, dict)
        self._outcome(
            context_row,
            label,
            damage=damage,
            self_damage=self_damage,
            spirit_cost=spirit_cost,
            duration=duration,
            startup=None,
            input_pattern=self._compress_trace(chain.input_trace),
            max_rows=128,
        )
        self.chains += 1
        self._emit(
            "human-chain",
            {
                "actions": chain.signatures,
                "context": chain.context,
                "damage": damage,
                "self_damage": self_damage,
                "spirit_cost": spirit_cost,
                "duration": duration,
            },
        )

    def observe(
        self,
        frame: int,
        state: BattleState,
        *,
        input_keys: frozenset[str] = frozenset(),
    ) -> None:
        me, enemy = state.p1, state.p2
        chord = self._normalized_chord(input_keys, int(getattr(me, "facing", 1)))
        self.input_history.append(chord)
        new_character = f"0x{me.vtable:08X}"
        new_opponent = f"0x{enemy.vtable:08X}"
        if (self.character_key, self.opponent_key) != (new_character, new_opponent):
            self._finish_action(frame)
            self._finish_chain(frame, None)
            self.input_history.clear()
            self.input_history.append(chord)
            self.character_key, self.opponent_key = new_character, new_opponent
        offensive = 300 <= me.action_id < 800
        chain_existed = self.chain is not None
        signature = f"{me.action_id}:{me.action_sequence}" if offensive else None
        if self.episode is not None:
            if enemy.hp < self.episode.enemy_hp and self.episode.first_hit_frame is None:
                self.episode.first_hit_frame = frame
            self.episode.enemy_hp = enemy.hp
            self.episode.me_hp = me.hp
            self.episode.spirit = int(me.spirit)
            if len(self.episode.input_trace) < 32:
                self.episode.input_trace.append(chord)
        if chain_existed and self.chain is not None and len(self.chain.input_trace) < 180:
            self.chain.input_trace.append(chord)
        if self.episode is not None and self.episode.signature != signature:
            self._finish_action(frame)
        if offensive and self.episode is None:
            self._begin_action(frame, state)
        if not offensive and self.chain is not None:
            self._finish_chain(frame, state)

    def end_encounter(self, frame: int, state: BattleState | None = None) -> None:
        self._finish_action(frame)
        self._finish_chain(frame, state)
        self.input_history.clear()

    def flush(self) -> None:
        self.data["updated_at"] = time.time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.output_path)

    def summary(self) -> dict[str, object]:
        return {
            "output": str(self.output_path),
            "action_episodes": self.action_episodes,
            "chains": self.chains,
            "matchups": len(self.data.get("matchups", {})),
        }
