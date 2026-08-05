#!/usr/bin/env python3
"""Summarize compact learning evidence for monitoring and snapshot comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th105.evaluation import evaluate_knowledge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("runtime/th105_opponent_models.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_knowledge(json.loads(args.knowledge.read_text(encoding="utf-8")))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
