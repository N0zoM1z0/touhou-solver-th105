#!/usr/bin/env python3
"""Validate and atomically install one distilled offline policy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.constants import EXPECTED_EXE_SHA256
from th105.offline_artifact import DistilledOutcomePolicy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install_bundle(
    bundle: Path,
    runtime_dir: Path,
    *,
    expected_game_build: str = EXPECTED_EXE_SHA256,
) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("offline bundle is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or int(manifest.get("artifact_schema_version", 0)) != 1:
        raise ValueError("unsupported offline bundle schema")
    distillation = manifest.get("distillation", {})
    if not isinstance(distillation, dict):
        raise ValueError("offline bundle has no distillation metadata")
    relative = str(distillation.get("file", ""))
    source = bundle / relative
    if not relative or not source.is_file() or source.parent != bundle:
        raise ValueError("distilled policy must be a direct file inside the bundle")
    actual_hash = sha256(source)
    if actual_hash != str(distillation.get("sha256", "")):
        raise ValueError("distilled policy hash mismatch")
    policy = DistilledOutcomePolicy.load(
        source, game_build_sha256=expected_game_build
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / "th105_offline_policy.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)
    installed_manifest = runtime_dir / "th105_offline_policy_manifest.json"
    manifest_temporary = installed_manifest.with_suffix(
        installed_manifest.suffix + ".tmp"
    )
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temporary, installed_manifest)
    return {
        "installed": destination.as_posix(),
        "sha256": actual_hash,
        "contexts": len(policy.contexts),
        "game_build_sha256": expected_game_build,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--expected-game-build", default=EXPECTED_EXE_SHA256)
    args = parser.parse_args()
    result = install_bundle(
        args.bundle,
        args.runtime_dir,
        expected_game_build=args.expected_game_build,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
