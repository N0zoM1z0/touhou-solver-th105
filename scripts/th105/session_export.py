"""Reproducible, privacy-bounded exports of one TH105 controller session."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from .menu import character_name
from .reward import DEFAULT_REWARD, REWARD_VERSION
from .schema import ACTION_SCHEMA_VERSION, TRAINING_GENERATION


EXPORT_SCHEMA_VERSION = 1
MODEL_FILES = (
    "th105_opponent_models.json",
    "th105_compiled_policy.json",
    "th105_human_demonstrations.json",
    "th105_human_policy.json",
)
SENSITIVE_FRAGMENTS = (
    "hf_",
    "/home/",
    "\\\\wsl.localhost",
    "password=",
    "sudo password",
)


def jsonl_family(path: Path) -> tuple[Path, ...]:
    """Return archived and bounded JSONL shards in oldest-to-newest order."""
    archive_dir = path.parent / "corpus_archive"
    archives = (
        sorted(
            archive_dir.glob(f"{path.name}.*.gz"),
            key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
        )
        if archive_dir.is_dir()
        else []
    )
    backups: list[tuple[int, Path]] = []
    for candidate in path.parent.glob(f"{path.name}.*.gz"):
        try:
            index = int(
                candidate.name.removeprefix(f"{path.name}.").removesuffix(".gz")
            )
        except ValueError:
            continue
        backups.append((index, candidate))
    ordered = archives + [
        candidate for _index, candidate in sorted(backups, reverse=True)
    ]
    if path.is_file():
        ordered.append(path)
    return tuple(ordered)


def iter_jsonl_family(path: Path) -> Iterator[dict[str, object]]:
    for source in jsonl_family(path):
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed JSONL in {source.name}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL in {source.name}:{line_number}")
                yield value


def _safe_json(value: object, *, label: str) -> bytes:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    lowered = encoded.decode("utf-8", errors="replace").casefold()
    for fragment in SENSITIVE_FRAGMENTS:
        if fragment.casefold() in lowered:
            raise ValueError(f"sensitive fragment {fragment!r} found in {label}")
    return encoded


def _write_jsonl_gz(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for index, record in enumerate(records):
                compressed.write(
                    _safe_json(record, label=f"{path.name}[{index}]") + b"\n"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_models(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        candidate = source / name
        if not candidate.is_file():
            continue
        raw = candidate.read_bytes()
        _safe_json(json.loads(raw), label=name)
        shutil.copy2(candidate, destination / name)


def _sanitize_session_events(
    records: Iterable[dict[str, object]],
    *,
    session_id: str,
    started_at: float,
    ended_at: float,
    difficulties: set[str],
    opponents: set[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        timestamp = float(record.get("time", 0.0))
        if timestamp < started_at or timestamp > ended_at:
            continue
        event = str(record.get("event", ""))
        difficulty = str(record.get("difficulty", ""))
        opponent = str(record.get("opponent", ""))
        if event == "encounter-start":
            if record.get("session_id") != session_id:
                continue
            if opponent not in opponents:
                continue
            output.append(
                {
                    "time": timestamp,
                    "event": event,
                    "session_id": session_id,
                    "difficulty": difficulty,
                    "opponent": record.get("opponent"),
                    "offline_policy_sha256": (
                        record.get("offline_artifact", {}).get("sha256")
                        if isinstance(record.get("offline_artifact"), dict)
                        else None
                    ),
                }
            )
        elif (
            event == "round-terminal"
            and difficulty in difficulties
            and opponent in opponents
        ):
            output.append(
                {
                    "time": timestamp,
                    "event": event,
                    "session_id": session_id,
                    "terminal_sequence": int(record.get("terminal_sequence", 0)),
                    "difficulty": difficulty,
                    "opponent": record.get("opponent"),
                    "won": record.get("won"),
                    "me_hp": int(record.get("me_hp", 0)),
                    "enemy_hp": int(record.get("enemy_hp", 0)),
                    "max_me_hp": int(record.get("max_me_hp", 0)),
                    "max_enemy_hp": int(record.get("max_enemy_hp", 0)),
                }
            )
    return output


def _dataset_card(manifest: dict[str, object]) -> str:
    experiment = manifest["experiment"]
    stats = manifest["statistics"]
    return f"""---
pretty_name: TH105 Autoplay Option Transitions
license: other
tags:
- touhou
- th105
- offline-rl
- contextual-bandit
- game-ai
---

# TH105 autoplay option-transition corpus

This dataset contains compact, event-aligned observations produced by
[touhou-solver-th105](https://github.com/N0zoM1z0/touhou-solver-th105).
It does not contain the game executable, game assets, screenshots, credentials,
or raw process-memory dumps.

This snapshot is experiment `{experiment["name"]}` from controller session
`{experiment["session_id"]}`. It contains {stats["transitions"]} option
transitions across {stats["terminal_rounds"]} native terminal rounds. The target
was {experiment["target_rounds"]} rounds; collection stops at a complete Arena
match boundary, so the actual count may be slightly higher.

The verified P1 is `{experiment["p1_character"]}`
(`{experiment["p1_vtable"]}`). Store this snapshot below
`{experiment["dataset_path_prefix"]}` in the Hugging Face dataset so artifacts
from different playable characters cannot share a checkpoint namespace.

## Files

- `data/transitions.jsonl.gz`: schema-versioned state/action/outcome/next-state
  records at policy-option boundaries.
- `data/terminals.jsonl.gz`: retained sanitized encounter/terminal summaries;
  this bounded stream may cover fewer rounds than the durable learning curve.
- `data/learning_curve.jsonl.gz`: one compact physical checkpoint per round.
- `models/baseline/`: online learner state before collection.
- `models/final/`: online learner state after collection.
- `manifest.json`: hashes, versions, reward scalarization, coverage, and counts.

Raw outcome components are retained separately from scalar reward. Offline
trainers can change reward weights without recollecting gameplay. Native hazard
and legal-action safety gates remain deployment authority; this corpus does not
license an offline model to bypass them.

## Limitations

- Behavior is adaptive. Action schema 4 records the complete current legal set;
  `legal_actions_known` and `behavior_probability` must still be honored.
- This is observational gameplay data, not randomized causal evidence.
- Offline metrics are diagnostic; complete physical matches remain the final
  policy evaluation.
- Touhou Project and Touhou Hisoutensoku/Scarlet Weather Rhapsody are properties
  of their respective rights holders. This project is unaffiliated.
"""


def export_session(
    *,
    runtime_dir: Path,
    output_dir: Path,
    session_id: str,
    started_at: float,
    ended_at: float,
    experiment_name: str,
    target_rounds: int,
    baseline_dir: Path | None = None,
    final_models_dir: Path | None = None,
    source_commit: str | None = None,
) -> dict[str, object]:
    if not session_id:
        raise ValueError("session id is required")
    if ended_at < started_at:
        raise ValueError("end time precedes start time")
    if target_rounds <= 0:
        raise ValueError("target rounds must be positive")

    transitions: list[dict[str, object]] = []
    seen: dict[str, bytes] = {}
    for record in iter_jsonl_family(runtime_dir / "th105_transitions.jsonl"):
        if record.get("session_id") != session_id:
            continue
        transition_id = str(record.get("transition_id", ""))
        if not transition_id:
            raise ValueError("session transition is missing transition_id")
        encoded = _safe_json(record, label=transition_id)
        prior = seen.get(transition_id)
        if prior is not None:
            if prior != encoded:
                raise ValueError(f"conflicting duplicate transition {transition_id}")
            continue
        seen[transition_id] = encoded
        transitions.append(record)
    if not transitions:
        raise ValueError(f"no transitions found for session {session_id}")
    action_schemas = {
        int(record.get("action_schema_version", 0)) for record in transitions
    }
    generations = {
        str(record.get("training_generation", "")) for record in transitions
    }
    if action_schemas != {ACTION_SCHEMA_VERSION}:
        raise ValueError(f"session has incompatible action schemas: {action_schemas}")
    if generations != {TRAINING_GENERATION}:
        raise ValueError(f"session has incompatible generations: {generations}")
    self_vtables = {
        str(record.get("state", {}).get("self", {}).get("character_vtable", ""))
        for record in transitions
        if isinstance(record.get("state"), dict)
        and isinstance(record.get("state", {}).get("self"), dict)
    }
    self_vtables.discard("")
    if len(self_vtables) != 1:
        raise ValueError(f"session mixes P1 character vtables: {sorted(self_vtables)}")
    p1_vtable = next(iter(self_vtables))
    try:
        p1_character = character_name(int(p1_vtable, 16))
    except ValueError:
        p1_character = None
    if p1_character is None:
        raise ValueError(f"unsupported P1 character vtable in session: {p1_vtable}")
    dataset_path_prefix = (
        f"characters/{p1_character}/{TRAINING_GENERATION}/experiments/"
        f"{experiment_name}"
    )

    difficulties = {str(record.get("difficulty", "")) for record in transitions}
    transition_opponents = {str(record.get("opponent", "")) for record in transitions}
    events = _sanitize_session_events(
        iter_jsonl_family(runtime_dir / "th105_live.jsonl"),
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        difficulties=difficulties,
        opponents=transition_opponents,
    )
    terminals = [event for event in events if event["event"] == "round-terminal"]
    curve_points = [
        record
        for record in iter_jsonl_family(runtime_dir / "th105_learning_curve.jsonl")
        if record.get("session_id") == session_id
        and started_at <= float(record.get("time", 0.0)) <= ended_at
    ]

    data_dir = output_dir / "data"
    _write_jsonl_gz(data_dir / "transitions.jsonl.gz", transitions)
    _write_jsonl_gz(data_dir / "terminals.jsonl.gz", events)
    _write_jsonl_gz(data_dir / "learning_curve.jsonl.gz", curve_points)
    if baseline_dir is not None:
        _copy_models(baseline_dir, output_dir / "models" / "baseline")
    _copy_models(
        final_models_dir or runtime_dir,
        output_dir / "models" / "final",
    )

    actions = Counter(str(record.get("action", "")) for record in transitions)
    opponents = sorted(transition_opponents)
    episodes = {str(record.get("episode_id", "")) for record in transitions}
    known_legal = sum(bool(record.get("legal_actions_known")) for record in transitions)
    legal_candidate_counts = [
        len(record.get("legal_actions", []))
        for record in transitions
        if isinstance(record.get("legal_actions"), list)
    ]
    punished = sum(
        bool(record.get("outcome", {}).get("punished")) for record in transitions
    )
    # Live telemetry is intentionally bounded and older rotations may no
    # longer be present in a long-running session.  The compact learning curve
    # is durable and has exactly one outcome checkpoint per native terminal
    # round, so it is the authority for physical round statistics.
    curve_outcomes = [
        record.get("outcome", {}).get("won")
        for record in curve_points
        if isinstance(record.get("outcome"), dict)
    ]
    terminal_outcomes = (
        curve_outcomes
        if curve_outcomes
        else [event.get("won") for event in terminals]
    )
    wins = sum(won is True for won in terminal_outcomes)
    losses = sum(won is False for won in terminal_outcomes)
    draws = len(terminal_outcomes) - wins - losses

    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "README.md"}:
            relative = path.relative_to(output_dir).as_posix()
            artifacts[relative] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }

    manifest: dict[str, object] = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "experiment": {
            "name": experiment_name,
            "session_id": session_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "target_rounds": target_rounds,
            "source_commit": source_commit,
            "p1_character": p1_character,
            "p1_vtable": p1_vtable,
            "dataset_path_prefix": dataset_path_prefix,
        },
        "schemas": {
            "transition": sorted(
                {int(record.get("schema_version", 0)) for record in transitions}
            ),
            "feature": sorted(
                {int(record.get("feature_schema_version", 0)) for record in transitions}
            ),
            "action": sorted(
                {int(record.get("action_schema_version", 0)) for record in transitions}
            ),
            "reward": REWARD_VERSION,
        },
        "game_build_sha256": sorted(
            {
                str(record.get("game_build_sha256"))
                for record in transitions
                if record.get("game_build_sha256")
            }
        ),
        "policy_sha256": sorted(
            {
                str(record.get("policy_sha256"))
                for record in transitions
                if record.get("policy_sha256")
            }
        ),
        "offline_policy_sha256": sorted(
            {
                str(record.get("offline_policy_sha256"))
                for record in transitions
                if record.get("offline_policy_sha256")
            }
        ),
        "reward": asdict(DEFAULT_REWARD),
        "statistics": {
            "transitions": len(transitions),
            "episodes": len(episodes),
            "terminal_rounds": len(terminal_outcomes),
            "retained_terminal_events": len(terminals),
            "learning_curve_points": len(curve_points),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "difficulties": sorted(difficulties),
            "opponents": opponents,
            "actions": dict(sorted(actions.items())),
            "legal_actions_known": known_legal,
            "legal_action_coverage": known_legal / len(transitions),
            "multi_action_transitions": sum(
                count > 1 for count in legal_candidate_counts
            ),
            "mean_legal_actions": (
                sum(legal_candidate_counts) / len(legal_candidate_counts)
                if legal_candidate_counts
                else 0.0
            ),
            "max_legal_actions": max(legal_candidate_counts, default=0),
            "transition_json_bytes": sum(
                len(_safe_json(record, label="transition-size")) + 1
                for record in transitions
            ),
            "transition_gzip_bytes": (data_dir / "transitions.jsonl.gz").stat().st_size,
            "gzip_bytes_per_transition": (
                (data_dir / "transitions.jsonl.gz").stat().st_size / len(transitions)
            ),
            "punished_transitions": punished,
            "punished_rate": punished / len(transitions),
        },
        "artifacts": artifacts,
        "privacy": {
            "sensitive_fragment_scan": "passed",
            "contains_game_binary": False,
            "contains_raw_memory_dump": False,
            "contains_credentials": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_bytes(
        _safe_json(manifest, label="manifest") + b"\n"
    )
    (output_dir / "README.md").write_text(_dataset_card(manifest), encoding="utf-8")
    return manifest
