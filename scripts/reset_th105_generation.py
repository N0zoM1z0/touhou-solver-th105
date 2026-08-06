#!/usr/bin/env python3
"""Purge legacy runtime evidence and initialize one clean learning generation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from th105.menu import CHARACTER_VTABLES
from th105.schema import (
    ACTION_SCHEMA_VERSION,
    CORPUS_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    TRAINING_GENERATION,
    TRANSITION_SCHEMA_VERSION,
)


PURGE_DIRECTORIES = (
    "corpus_archive",
    "experiments",
    "policy-validation",
    "policy-zoo",
)
PURGE_FILE_PREFIXES = ("th105_",)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _purge_runtime(runtime_dir: Path) -> tuple[int, int]:
    files = 0
    bytes_removed = 0
    for name in PURGE_DIRECTORIES:
        target = runtime_dir / name
        if not target.exists():
            continue
        bytes_removed += sum(
            path.stat().st_size for path in target.rglob("*") if path.is_file()
        )
        files += sum(1 for path in target.rglob("*") if path.is_file())
        shutil.rmtree(target)
    for target in tuple(runtime_dir.iterdir()) if runtime_dir.is_dir() else ():
        if not target.is_file() or not target.name.startswith(PURGE_FILE_PREFIXES):
            continue
        bytes_removed += target.stat().st_size
        files += 1
        target.unlink()
    return files, bytes_removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument(
        "--p1-character", required=True, choices=tuple(CHARACTER_VTABLES)
    )
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--purge-existing", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    runtime_dir = args.runtime_dir.resolve()
    expected_runtime = (repository / "runtime").resolve()
    if runtime_dir != expected_runtime:
        raise ValueError(f"refusing non-workspace runtime path: {runtime_dir}")
    if not args.purge_existing:
        raise ValueError("clean generation initialization requires --purge-existing")

    runtime_dir.mkdir(parents=True, exist_ok=True)
    files, bytes_removed = _purge_runtime(runtime_dir)
    started_at = time.time()
    knowledge = {
        "version": 1,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "training_generation": TRAINING_GENERATION,
        "characters": {},
    }
    generation = {
        "manifest_schema_version": 1,
        "training_generation": TRAINING_GENERATION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "started_at": started_at,
        "source_commit": args.source_commit,
        "dataset_repo": args.dataset_repo,
        "dataset_path_prefix": (
            f"characters/{args.p1_character}/{TRAINING_GENERATION}"
        ),
        "p1_character": args.p1_character,
        "p1_vtable": f"0x{CHARACTER_VTABLES[args.p1_character]:08X}",
        "game_build_sha256": (
            "49c23d9467b9927ba687ed2b873c4bc2d2f39ddadc9f55051ccf10172c0b7c11"
        ),
        "schemas": {
            "corpus": CORPUS_SCHEMA_VERSION,
            "transition": TRANSITION_SCHEMA_VERSION,
            "feature": FEATURE_SCHEMA_VERSION,
            "action": ACTION_SCHEMA_VERSION,
        },
        "baseline": {
            "rounds": 0,
            "transitions": 0,
            "opponents": 0,
            "human_demonstrations": False,
            "offline_policy": False,
            "cancel_edges": 0,
            "offense_trials": 0,
        },
        "legacy_data": {
            "online_reuse": False,
            "offline_reuse": False,
            "recoverable_hf_dataset": (
                "Joh1rreq/touhou-solver-th105-autonomous-corpus"
            ),
            "recoverable_hf_revision": (
                "90d4994c5f8000e7ceceaf9a4ec01333bdf5b136"
            ),
        },
    }
    _write_json(runtime_dir / "th105_opponent_models.json", knowledge)
    _write_json(runtime_dir / "th105_generation.json", generation)

    if args.dataset_output is not None:
        output = args.dataset_output.resolve()
        generation_output = output / str(generation["dataset_path_prefix"])
        if generation_output.exists():
            raise ValueError(
                f"dataset generation output already exists: {generation_output}"
            )
        (generation_output / "baseline").mkdir(parents=True)
        shutil.copy2(
            runtime_dir / "th105_opponent_models.json",
            generation_output / "baseline" / "th105_opponent_models.json",
        )
        shutil.copy2(
            runtime_dir / "th105_generation.json",
            generation_output / "generation.json",
        )
        shutil.copy2(
            repository / "notes" / "HF_AUTONOMOUS_CORPUS_README.md",
            generation_output / "README.md",
        )
        if not (output / "README.md").exists():
            shutil.copy2(
                repository / "notes" / "HF_AUTONOMOUS_CORPUS_README.md",
                output / "README.md",
            )
    print(
        json.dumps(
            {
                "training_generation": TRAINING_GENERATION,
                "removed_files": files,
                "removed_bytes": bytes_removed,
                "runtime_dir": str(runtime_dir),
                "p1_character": args.p1_character,
                "dataset_path_prefix": generation["dataset_path_prefix"],
                "dataset_output": (
                    str(generation_output) if args.dataset_output else None
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
