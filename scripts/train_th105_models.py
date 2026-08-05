#!/usr/bin/env python3
"""Compile accumulated TH105 experience into a compact online lookup table."""

from __future__ import annotations

import argparse
from pathlib import Path

from th105.model_compiler import compile_knowledge_file


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
    args = parser.parse_args()
    compiled = compile_knowledge_file(args.knowledge, args.output)
    characters = compiled.get("characters", {})
    print(f"compiled {len(characters)} character models -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
