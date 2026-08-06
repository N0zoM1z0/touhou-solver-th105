#!/usr/bin/env python3
"""Continuously preserve bounded TH105 transition rotations as immutable shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from th105.telemetry import archive_gzip_shard


def archive_once(runtime_dir: Path, archive_dir: Path) -> dict[str, int]:
    source = runtime_dir / "th105_transitions.jsonl"
    archived = 0
    skipped = 0
    raced = 0
    candidates: list[tuple[int, Path]] = []
    for candidate in runtime_dir.glob(f"{source.name}.*.gz"):
        try:
            index = int(
                candidate.name.removeprefix(f"{source.name}.").removesuffix(".gz")
            )
        except ValueError:
            continue
        candidates.append((index, candidate))
    for _index, candidate in sorted(candidates, reverse=True):
        try:
            before = set(archive_dir.glob(f"{source.name}.*.gz"))
            destination = archive_gzip_shard(
                candidate,
                archive_dir,
                logical_name=source.name,
            )
        except FileNotFoundError:
            raced += 1
            continue
        if destination in before:
            skipped += 1
        else:
            archived += 1
    return {"archived": archived, "skipped": skipped, "raced": raced}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    runtime_dir = args.runtime_dir.resolve()
    archive_dir = (
        args.archive_dir.resolve()
        if args.archive_dir is not None
        else runtime_dir / "corpus_archive"
    )
    while True:
        result = archive_once(runtime_dir, archive_dir)
        result.update({"archive_dir": str(archive_dir), "time": time.time()})
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
