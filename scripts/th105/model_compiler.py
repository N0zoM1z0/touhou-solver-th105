"""Compile cumulative combat knowledge into a compact read-mostly artifact."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .defense_learning import DefenseResponseModel


DEFENSE_ORDER = ("high_guard", "low_guard", "backdash", "jump_back")


def compile_knowledge(data: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "version": 1,
        "generated_at": time.time(),
        "characters": {},
    }
    source_characters = data.get("characters", {})
    if not isinstance(source_characters, dict):
        return output
    compiled_characters = output["characters"]
    assert isinstance(compiled_characters, dict)
    for character_key, raw_entry in source_characters.items():
        if not isinstance(raw_entry, dict):
            continue
        defense_state = raw_entry.get("defense_responses", {})
        defense = DefenseResponseModel()
        defense.import_state(defense_state)
        choices: dict[str, object] = {}
        for signature, row in defense.table.items():
            response = defense.choose(signature, DEFENSE_ORDER)
            selected = row[response]
            total_trials = sum(item.trials for item in row.values())
            choices[signature] = {
                "response": response,
                "trials": total_trials,
                "selected_trials": selected.trials,
                "successes": selected.successes,
                "damage_events": selected.damage_events,
                "mean_damage": selected.mean_damage,
            }

        temporal: dict[str, object] = {}
        raw_profiles = raw_entry.get("profiles", {})
        if isinstance(raw_profiles, dict):
            for action_id, profile in raw_profiles.items():
                if not isinstance(profile, dict):
                    continue
                markers = [
                    float(profile[name])
                    for name, sample_name in (
                        ("first_impact", "impact_samples"),
                        ("first_projectile", "projectile_samples"),
                        ("first_active_box", "active_box_observations"),
                    )
                    if int(profile.get(sample_name, 0)) > 0
                ]
                duration = float(profile.get("mean_total", 0.0))
                temporal[str(action_id)] = {
                    "danger_start": min(markers) if markers else None,
                    "duration": duration,
                    "samples": int(profile.get("completions", 0)),
                }

        projectile_state = raw_entry.get("projectile_envelopes", {})
        projectile_extents = (
            projectile_state.get("extents", {})
            if isinstance(projectile_state, dict) else {}
        )
        offense_state = raw_entry.get("offense_outcomes", {})
        offense_choices: dict[str, object] = {}
        if isinstance(offense_state, dict):
            for context, raw_row in offense_state.items():
                if not isinstance(raw_row, dict):
                    continue
                ranked: list[tuple[float, str, dict[str, object]]] = []
                for label, raw in raw_row.items():
                    if not isinstance(raw, dict):
                        continue
                    trials = max(1, int(raw.get("trials", 0)))
                    utility = (
                        float(raw.get("total_damage", 0))
                        - 1.35 * float(raw.get("total_self_damage", 0))
                        - 0.20 * float(raw.get("total_spirit_cost", 0))
                        - 2.0 * float(raw.get("total_commitment", 0))
                    ) / trials
                    ranked.append((utility, str(label), raw))
                if ranked:
                    utility, label, raw = max(ranked)
                    offense_choices[str(context)] = {
                        "action": label,
                        "utility": utility,
                        "trials": int(raw.get("trials", 0)),
                        "connections": int(raw.get("connections", 0)),
                    }
        compiled_characters[str(character_key)] = {
            "defense_choices": choices,
            "projectile_extents": projectile_extents,
            "temporal_actions": temporal,
            "offense_choices": offense_choices,
        }
    return output


def compile_knowledge_file(source: Path, destination: Path) -> dict[str, object]:
    data = json.loads(source.read_text(encoding="utf-8"))
    compiled = compile_knowledge(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return compiled
