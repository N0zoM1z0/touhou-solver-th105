#!/usr/bin/env python3
"""Collect a fixed-difficulty/opponent corpus from one frozen online baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_th105_policy_ab import (
    DEFAULT_WINDOWS_PYTHON,
    controller_result,
    copy_models,
    restore_models,
    stop_game,
)
from th105.menu import CHARACTERS, DIFFICULTIES
from th105.session_export import export_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible physical collection: fixed CPU difficulty, "
            "one recorded opponent, and an immutable starting online checkpoint."
        )
    )
    parser.add_argument("--difficulty", choices=DIFFICULTIES, required=True)
    parser.add_argument("--opponent", choices=CHARACTERS, required=True)
    parser.add_argument(
        "--arena-rounds",
        type=int,
        required=True,
        help="total Arena rounds to run; only the requested opponent is exported",
    )
    parser.add_argument("--exploration-rate", type=float, default=0.16)
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-python", type=Path, default=DEFAULT_WINDOWS_PYTHON)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()
    if args.arena_rounds <= 0:
        parser.error("--arena-rounds must be positive")
    if not 0.0 <= args.exploration_rate <= 1.0:
        parser.error("--exploration-rate must be between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must not already contain files")
    if not args.windows_python.is_file():
        parser.error("Windows Python executable was not found")

    repository = Path(__file__).resolve().parents[1]
    runtime = args.runtime_dir.resolve()
    output = args.output.resolve()
    baseline = output / "baseline"
    copy_models(runtime, baseline)
    if not (baseline / "th105_opponent_models.json").is_file():
        parser.error("runtime has no online opponent-model baseline")

    command = [
        str(args.windows_python),
        "scripts/run_th105_agent.py",
        "auto-arcade",
        "--launch",
        "--p1-character",
        "sakuya",
        "--difficulty",
        args.difficulty,
        "--continuous",
        "--round-limit",
        str(args.arena_rounds),
        "--collect-opponent",
        args.opponent,
        "--freeze-online-checkpoint",
        "--exploration-rate",
        str(args.exploration_rate),
        "--policy",
        "adaptive",
        "--timeout",
        str(args.timeout),
    ]
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "controller.json"
    stderr_path = output / "controller.stderr.log"
    started_at = time.time()
    stop_game()
    try:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=repository,
                stdout=stdout,
                stderr=stderr,
                check=False,
                env=environment,
            )
        ended_at = time.time()
        stop_game()
        if completed.returncode != 0:
            raise RuntimeError(f"controller exited with {completed.returncode}")
        result = controller_result(stdout_path)
        fight = result["fight"]
        assert isinstance(fight, dict)
        manifest = export_session(
            runtime_dir=runtime,
            output_dir=output / "session",
            session_id=str(fight["session_id"]),
            started_at=started_at,
            ended_at=ended_at,
            experiment_name=(
                f"fixed-{args.difficulty}-{args.opponent}-coverage-collection"
            ),
            target_rounds=args.arena_rounds,
            baseline_dir=baseline,
            final_models_dir=baseline,
        )
        summary = {
            "schema_version": 1,
            "difficulty": args.difficulty,
            "opponent": args.opponent,
            "arena_rounds": args.arena_rounds,
            "exploration_rate": args.exploration_rate,
            "online_checkpoint_frozen": True,
            "session_id": fight["session_id"],
            "statistics": manifest["statistics"],
        }
        (output / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    finally:
        stop_game()
        restore_models(baseline, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
