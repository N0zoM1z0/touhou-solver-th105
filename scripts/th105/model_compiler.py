"""Compile cumulative combat knowledge into a compact read-mostly artifact."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .defense_learning import DefenseResponseModel
from .offense_learning import ActionOutcomeModel
from .reward import REWARD_VERSION, empirical_action_value


DEFENSE_ORDER = ("high_guard", "low_guard", "backdash", "jump_back")


def compile_knowledge(data: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "version": 1,
        "source_schema_version": int(data.get("schema_version", 1)),
        "reward_version": REWARD_VERSION,
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
            offense = ActionOutcomeModel()
            offense.import_state(offense_state)
            for context, row in offense.table.items():
                ranked = [
                    (empirical_action_value(stats), label, stats)
                    for label, stats in row.items()
                ]
                if ranked:
                    utility, label, stats = max(ranked)
                    offense_choices[str(context)] = {
                        "action": label,
                        "utility": utility,
                        "trials": stats.trials,
                        "connections": stats.connections,
                        "reward_version": REWARD_VERSION,
                    }
        geometry_state = raw_entry.get("attack_geometry", {})
        attack_geometry: dict[str, object] = {}
        if isinstance(geometry_state, dict):
            for signature, raw in geometry_state.items():
                if not isinstance(raw, dict):
                    continue
                envelope = raw.get("attack_envelope")
                if not isinstance(envelope, list) or len(envelope) != 4:
                    continue
                attack_geometry[str(signature)] = {
                    "first_active": int(raw.get("first_active_elapsed", 0)),
                    "last_active": int(raw.get("last_active_elapsed", 0)),
                    "envelope": [int(value) for value in envelope],
                    "samples": int(raw.get("active_observations", 0)),
                    "poses": len(raw.get("poses", {}))
                    if isinstance(raw.get("poses"), dict) else 0,
                }
        cancel_state = raw_entry.get("cancel_graph", {})
        cancel_edges: dict[str, object] = {}
        if isinstance(cancel_state, dict):
            ranked_edges = sorted(
                (
                    (str(edge_id), raw)
                    for edge_id, raw in cancel_state.items()
                    if isinstance(raw, dict)
                ),
                key=lambda item: (
                    int(item[1].get("direct_damage_events", 0)),
                    int(item[1].get("trials", 0)),
                    item[0],
                ),
                reverse=True,
            )[:128]
            for edge_id, raw in ranked_edges:
                cancel_edges[edge_id] = {
                    "target": [
                        int(raw.get("target_action", 0)),
                        int(raw.get("target_sequence", 0)),
                    ],
                    "chord": str(raw.get("chord", "")),
                    "frame_window": [
                        int(raw.get("minimum_source_frame", 0)),
                        int(raw.get("maximum_source_frame", 0)),
                    ],
                    "trials": int(raw.get("trials", 0)),
                    "direct_damage_events": int(raw.get("direct_damage_events", 0)),
                }
        compiled_characters[str(character_key)] = {
            "defense_choices": choices,
            "projectile_extents": projectile_extents,
            "temporal_actions": temporal,
            "offense_choices": offense_choices,
            "attack_geometry": attack_geometry,
            "cancel_edges": cancel_edges,
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
