#!/usr/bin/env python3
"""Compile accumulated TH105 experience into a compact online lookup table."""

from __future__ import annotations

import argparse
from pathlib import Path

from th105.model_compiler import compile_knowledge_file
from th105.human_learning import compile_human_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("runtime/th105_opponent_models.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/th105_compiled_policy.json"),
    )
    parser.add_argument(
        "--human-demonstrations",
        type=Path,
        default=Path("runtime/th105_human_demonstrations.json"),
    )
    parser.add_argument(
        "--human-output",
        type=Path,
        default=Path("runtime/th105_human_policy.json"),
    )
    args = parser.parse_args()
    compiled = compile_knowledge_file(args.knowledge, args.output)
    characters = compiled.get("characters", {})
    print(f"compiled {len(characters)} character models -> {args.output}")
    if args.human_demonstrations.is_file():
        human = compile_human_file(args.human_demonstrations, args.human_output)
        matchups = human.get("matchups", {})
        print(f"compiled {len(matchups)} human matchup priors -> {args.human_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
