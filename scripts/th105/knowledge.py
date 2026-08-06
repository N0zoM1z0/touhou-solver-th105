"""Atomic persistence for per-character temporal action knowledge."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .schema import ACTION_SCHEMA_VERSION, CORPUS_SCHEMA_VERSION, TRAINING_GENERATION
MODEL_FIELDS = (
    "profiles",
    "projectile_envelopes",
    "defense_responses",
    "offense_outcomes",
    "attack_geometry",
    "cancel_graph",
)


def load_knowledge(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "version": 1,
            "schema_version": CORPUS_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "training_generation": TRAINING_GENERATION,
            "characters": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("version") != 1
        or int(data.get("schema_version", 1)) != CORPUS_SCHEMA_VERSION
        or int(data.get("action_schema_version", 0)) != ACTION_SCHEMA_VERSION
        or str(data.get("training_generation", "")) != TRAINING_GENERATION
        or not isinstance(data.get("characters"), dict)
    ):
        raise ValueError(f"unsupported opponent knowledge format: {path}")
    return data


def character_models_from_data(
    data: object, character_key: str
) -> dict[str, dict[str, object]]:
    """Extract every model family from one already-parsed corpus snapshot."""
    if not isinstance(data, dict):
        return {name: {} for name in MODEL_FIELDS}
    characters = data.get("characters", {})
    entry = characters.get(character_key, {}) if isinstance(characters, dict) else {}
    if not isinstance(entry, dict):
        return {name: {} for name in MODEL_FIELDS}
    return {
        name: value if isinstance((value := entry.get(name, {})), dict) else {}
        for name in MODEL_FIELDS
    }


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
    data["action_schema_version"] = ACTION_SCHEMA_VERSION
    data["training_generation"] = TRAINING_GENERATION
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
