"""Atomic persistence for per-character temporal action knowledge."""

from __future__ import annotations

import json
import time
from pathlib import Path

CORPUS_SCHEMA_VERSION = 1


def load_knowledge(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "version": 1,
            "schema_version": CORPUS_SCHEMA_VERSION,
            "characters": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("version") != 1
        or int(data.get("schema_version", 1)) != CORPUS_SCHEMA_VERSION
        or not isinstance(data.get("characters"), dict)
    ):
        raise ValueError(f"unsupported opponent knowledge format: {path}")
    return data


def profiles_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    profiles = entry.get("profiles", {}) if isinstance(entry, dict) else {}
    return profiles if isinstance(profiles, dict) else {}


def projectile_model_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    model = entry.get("projectile_envelopes", {}) if isinstance(entry, dict) else {}
    return model if isinstance(model, dict) else {}


def defense_model_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    model = entry.get("defense_responses", {}) if isinstance(entry, dict) else {}
    return model if isinstance(model, dict) else {}


def offense_model_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    model = entry.get("offense_outcomes", {}) if isinstance(entry, dict) else {}
    return model if isinstance(model, dict) else {}


def attack_geometry_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    model = entry.get("attack_geometry", {}) if isinstance(entry, dict) else {}
    return model if isinstance(model, dict) else {}


def cancel_graph_for(path: Path, character_key: str) -> dict[str, object]:
    data = load_knowledge(path)
    entry = data["characters"].get(character_key, {})
    model = entry.get("cancel_graph", {}) if isinstance(entry, dict) else {}
    return model if isinstance(model, dict) else {}


def persist_profiles(
    path: Path, character_key: str, profiles: dict[str, object]
) -> None:
    persist_character_models(path, character_key, profiles=profiles)


def persist_character_models(
    path: Path,
    character_key: str,
    *,
    profiles: dict[str, object],
    projectile_envelopes: dict[str, object] | None = None,
    defense_responses: dict[str, object] | None = None,
    offense_outcomes: dict[str, object] | None = None,
    attack_geometry: dict[str, object] | None = None,
    cancel_graph: dict[str, object] | None = None,
) -> None:
    data = load_knowledge(path)
    data["schema_version"] = CORPUS_SCHEMA_VERSION
    characters = data["characters"]
    previous = characters.get(character_key, {})
    entry = dict(previous) if isinstance(previous, dict) else {}
    entry.update({"updated_at": time.time(), "profiles": profiles})
    if projectile_envelopes is not None:
        entry["projectile_envelopes"] = projectile_envelopes
    if defense_responses is not None:
        entry["defense_responses"] = defense_responses
    if offense_outcomes is not None:
        entry["offense_outcomes"] = offense_outcomes
    if attack_geometry is not None:
        entry["attack_geometry"] = attack_geometry
    if cancel_graph is not None:
        entry["cancel_graph"] = cancel_graph
    characters[character_key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
