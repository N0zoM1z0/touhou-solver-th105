#!/usr/bin/env python3
"""Export one controller session as a portable Hugging Face dataset snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.session_export import export_session


def _source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--started-at", required=True, type=float)
    parser.add_argument("--ended-at", required=True, type=float)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--target-rounds", required=True, type=int)
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_session(
        runtime_dir=args.runtime_dir,
        output_dir=args.output,
        session_id=args.session_id,
        started_at=args.started_at,
        ended_at=args.ended_at,
        experiment_name=args.experiment_name,
        target_rounds=args.target_rounds,
        baseline_dir=args.baseline_dir,
        source_commit=_source_commit(),
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
