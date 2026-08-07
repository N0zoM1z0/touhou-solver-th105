#!/usr/bin/env python3
"""Replay reusable TH105 JSONL shards into v6 online model priors."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.knowledge import load_knowledge
from th105.offline_replay import replay_transitions
from th105.schema import (
    ACTION_SCHEMA_VERSION,
    TRAINING_GENERATION,
    TRANSITION_SCHEMA_VERSION,
)


def _records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{line_number}: transition is not an object"
                    )
                if int(record.get("schema_version", 0)) != TRANSITION_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: incompatible transition schema"
                    )
                if int(record.get("action_schema_version", 0)) != ACTION_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: incompatible action schema"
                    )
                if str(record.get("training_generation", "")) != TRAINING_GENERATION:
                    raise ValueError(f"{path}:{line_number}: incompatible generation")
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--knowledge", type=Path, default=Path("runtime/th105_opponent_models.json")
    )
    parser.add_argument("--self-character", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runtime/th105_replayed_models.json")
    )
    args = parser.parse_args()
    knowledge = load_knowledge(args.knowledge)
    records = _records(args.inputs)
    result = replay_transitions(records, knowledge, self_character=args.self_character)
    characters = knowledge["characters"]
    assert isinstance(characters, dict)
    for opponent, models in result.opponents.items():
        raw_entry = characters.get(opponent, {})
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {}
        entry.update(models)
        entry["offline_replay_updated_at"] = time.time()
        characters[opponent] = entry
    knowledge["offline_replay"] = {
        "version": 1,
        "policy": "enemy-action-options-v6",
        "self_character": args.self_character,
        "transitions_seen": result.transitions_seen,
        "transitions_used": result.transitions_used,
        "offense_samples": result.offense_samples,
        "option_samples": result.option_samples,
        "opponents": len(result.opponents),
        "source_files": [str(path) for path in args.inputs],
        "generated_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(knowledge, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(knowledge["offline_replay"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
