#!/usr/bin/env python3
"""Compare held-out metrics and distilled decisions across policy bundles."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.offline_artifact import predicted_outcome_utility


def _load_bundle(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    distilled = json.loads(
        (path / str(manifest["distillation"]["file"])).read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict) or not isinstance(distilled, dict):
        raise ValueError(f"invalid policy bundle: {path}")
    return manifest, distilled


def _top_actions(distilled: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    contexts = distilled.get("contexts", {})
    if not isinstance(contexts, dict):
        return result
    for context, actions in contexts.items():
        if not isinstance(actions, dict) or len(actions) < 2:
            continue
        scored: list[tuple[float, str]] = []
        for action, entry in actions.items():
            if not isinstance(entry, dict):
                continue
            outcomes = entry.get("outcomes", {})
            if isinstance(outcomes, dict):
                scored.append((predicted_outcome_utility(outcomes), str(action)))
        if scored:
            result[str(context)] = max(scored, key=lambda item: (item[0], item[1]))[1]
    return result


def _coverage_summary(distilled: dict[str, object]) -> dict[str, int | float]:
    contexts = distilled.get("contexts", {})
    if not isinstance(contexts, dict):
        return {
            "contexts": 0,
            "context_actions": 0,
            "multi_action_contexts": 0,
            "counterfactual_context_actions": 0,
            "mean_actions_per_context": 0.0,
        }
    action_count = 0
    multi = 0
    counterfactual = 0
    for actions in contexts.values():
        if not isinstance(actions, dict):
            continue
        action_count += len(actions)
        multi += len(actions) >= 2
        for entry in actions.values():
            if not isinstance(entry, dict):
                continue
            if int(entry.get("factual_support", entry.get("support", 0))) < int(
                entry.get("support", 0)
            ):
                counterfactual += 1
    return {
        "contexts": len(contexts),
        "context_actions": action_count,
        "multi_action_contexts": multi,
        "counterfactual_context_actions": counterfactual,
        "mean_actions_per_context": action_count / len(contexts) if contexts else 0.0,
    }


def compare_distilled(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    left_top = _top_actions(left)
    right_top = _top_actions(right)
    common = sorted(set(left_top) & set(right_top))
    agreements = sum(left_top[context] == right_top[context] for context in common)
    return {
        "common_multi_action_contexts": len(common),
        "top_action_agreements": agreements,
        "top_action_disagreements": len(common) - agreements,
        "top_action_agreement_rate": agreements / len(common) if common else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.bundles) < 2:
        parser.error("at least two bundles are required")

    loaded: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    source_hashes = set()
    summaries: dict[str, object] = {}
    for bundle in args.bundles:
        manifest, distilled = _load_bundle(bundle)
        name = bundle.name
        if name in loaded:
            raise ValueError(f"duplicate bundle name: {name}")
        loaded[name] = (manifest, distilled)
        source = manifest.get("source", {})
        if isinstance(source, dict):
            source_hashes.add(str(source.get("transitions_sha256")))
        heads = manifest.get("heads", {})
        summaries[name] = {
            "kind": manifest.get("kind"),
            "distillation": manifest.get("distillation"),
            "candidate_coverage": _coverage_summary(distilled),
            "held_out": {
                head: metadata.get("validation")
                for head, metadata in heads.items()
                if isinstance(metadata, dict)
            }
            if isinstance(heads, dict)
            else {},
        }
    if len(source_hashes) != 1:
        raise ValueError("policy bundles were not trained from the same transitions")

    pairwise: dict[str, object] = {}
    for left_name, right_name in itertools.combinations(sorted(loaded), 2):
        pairwise[f"{left_name}__vs__{right_name}"] = compare_distilled(
            loaded[left_name][1], loaded[right_name][1]
        )
    result = {
        "transitions_sha256": next(iter(source_hashes)),
        "bundles": summaries,
        "pairwise": pairwise,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
