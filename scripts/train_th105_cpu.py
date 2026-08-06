#!/usr/bin/env python3
"""Train CPU-only tabular outcome heads and a compact distilled policy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.offline_cpu import (
    CATEGORICAL_FEATURES,
    CPU_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    candidate_prediction_records,
    distillation_context,
    feature_vector,
    outcome_targets,
    temporal_episode_split,
    validate_transition_schemas,
)
from th105.reward import DEFAULT_REWARD, REWARD_VERSION
from dataclasses import asdict


ARTIFACT_SCHEMA_VERSION = 1


def _load_records(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed transition at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object transition at line {line_number}")
            records.append(value)
    if len(records) < 2:
        raise ValueError("training requires at least two transitions")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    return {
        "mae": sum(abs(error) for error in errors) / max(1, len(errors)),
        "rmse": math.sqrt(sum(error * error for error in errors) / max(1, len(errors))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-distill-support", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0 or args.iterations <= 0 or args.min_distill_support <= 0:
        parser.error("threads, iterations, and distillation support must be positive")

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise SystemExit(
            "CatBoost is required on the CPU training server; install "
            "requirements-cpu-train.txt"
        ) from exc

    records = _load_records(args.input)
    validate_transition_schemas(records)
    corpus_manifest: dict[str, object] = {}
    if args.corpus_manifest and args.corpus_manifest.is_file():
        loaded_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, dict):
            raise ValueError("corpus manifest must be a JSON object")
        corpus_manifest = loaded_manifest
    prediction_records = candidate_prediction_records(records)
    rows = [feature_vector(record) for record in prediction_records]
    targets = [outcome_targets(record) for record in records]
    train_indices, validation_indices = temporal_episode_split(
        records, validation_fraction=args.validation_fraction
    )
    categorical_indices = [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
    train_rows = [rows[index] for index in train_indices]
    validation_rows = [rows[index] for index in validation_indices]
    args.output.mkdir(parents=True, exist_ok=True)
    models_dir = args.output / "models"
    models_dir.mkdir(exist_ok=True)

    head_specs: tuple[tuple[str, str, Callable[[dict[str, float]], float]], ...] = (
        ("damage_bp", "RMSE", lambda target: target["damage_bp"]),
        (
            "connection_probability",
            "RMSE",
            lambda target: target["connection_probability"],
        ),
        ("self_damage_bp", "RMSE", lambda target: target["self_damage_bp"]),
        (
            "self_damage_p90_bp",
            "Quantile:alpha=0.9",
            lambda target: target["self_damage_bp"],
        ),
        ("spirit_cost_bp", "RMSE", lambda target: target["spirit_cost_bp"]),
        ("punished_probability", "RMSE", lambda target: target["punished_probability"]),
        ("commitment_frames", "RMSE", lambda target: target["commitment_frames"]),
        ("terminal_value", "RMSE", lambda target: target["terminal_value"]),
    )
    predictions: dict[str, list[float]] = {}
    heads: dict[str, dict[str, object]] = {}
    for index, (name, loss, target_fn) in enumerate(head_specs):
        values = [float(target_fn(target)) for target in targets]
        train_values = [values[row] for row in train_indices]
        validation_values = [values[row] for row in validation_indices]
        if max(train_values) == min(train_values):
            constant = train_values[0]
            predicted_all = [constant] * len(records)
            heads[name] = {
                "kind": "constant",
                "value": constant,
                "validation": _metrics(
                    validation_values, [constant] * len(validation_values)
                ),
            }
        else:
            model = CatBoostRegressor(
                iterations=args.iterations,
                depth=args.depth,
                learning_rate=args.learning_rate,
                loss_function=loss,
                random_seed=10500 + index,
                thread_count=args.threads,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(
                train_rows,
                train_values,
                cat_features=categorical_indices,
                eval_set=(validation_rows, validation_values),
                early_stopping_rounds=max(20, args.iterations // 10),
                verbose=False,
            )
            predicted_all = [float(value) for value in model.predict(rows)]
            model_path = models_dir / f"{name}.cbm"
            model.save_model(model_path)
            heads[name] = {
                "kind": "catboost-regressor",
                "loss": loss,
                "file": model_path.relative_to(args.output).as_posix(),
                "sha256": _sha256(model_path),
                "best_iteration": int(model.get_best_iteration()),
                "validation": _metrics(
                    validation_values,
                    [predicted_all[row] for row in validation_indices],
                ),
            }
        predictions[name] = predicted_all

    aggregates: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    working: dict[tuple[str, str], dict[str, object]] = {}
    for row_index, record in enumerate(prediction_records):
        context = distillation_context(record)
        action = str(record.get("action", "unknown"))
        key = (context, action)
        entry = working.setdefault(
            key,
            {
                "support": 0,
                "factual_support": 0,
                "sums": {name: 0.0 for name in predictions},
            },
        )
        entry["support"] = int(entry["support"]) + 1
        if action == str(record.get("__factual_action", action)):
            entry["factual_support"] = int(entry["factual_support"]) + 1
        sums = entry["sums"]
        assert isinstance(sums, dict)
        for name, values in predictions.items():
            sums[name] = float(sums[name]) + values[row_index]
    for (context, action), entry in working.items():
        support = int(entry["support"])
        if support < args.min_distill_support:
            continue
        sums = entry["sums"]
        assert isinstance(sums, dict)
        aggregates[context][action] = {
            "support": support,
            "factual_support": int(entry["factual_support"]),
            "prediction_kind": (
                "factual-and-counterfactual"
                if int(entry["factual_support"]) < support
                else "factual"
            ),
            "outcomes": {
                name: float(total) / support for name, total in sorted(sums.items())
            },
        }
    distilled = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "context_schema_version": 1,
        "outcome_heads": sorted(predictions),
        "compatibility": {
            "game_build_sha256": corpus_manifest.get("game_build_sha256", []),
            "difficulties": (
                corpus_manifest.get("statistics", {}).get("difficulties", [])
                if isinstance(corpus_manifest.get("statistics"), dict)
                else []
            ),
            "opponents": (
                corpus_manifest.get("statistics", {}).get("opponents", [])
                if isinstance(corpus_manifest.get("statistics"), dict)
                else []
            ),
        },
        "contexts": {
            context: actions for context, actions in sorted(aggregates.items())
        },
    }
    distilled_path = args.output / "distilled_policy.json"
    distilled_path.write_text(
        json.dumps(distilled, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": "catboost-cpu-multihead-with-distillation",
        "trainer": {
            "task_type": "CPU",
            "threads": args.threads,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        },
        "source": {
            "transitions_sha256": _sha256(args.input),
            "corpus_manifest_sha256": (
                _sha256(args.corpus_manifest)
                if args.corpus_manifest and args.corpus_manifest.is_file()
                else None
            ),
            "records": len(records),
            "candidate_prediction_rows": len(prediction_records),
            "counterfactual_prediction_rows": len(prediction_records) - len(records),
            "train_records": len(train_indices),
            "validation_records": len(validation_indices),
        },
        "schemas": {
            "cpu_feature": CPU_FEATURE_SCHEMA_VERSION,
            "reward": REWARD_VERSION,
        },
        "features": {
            "names": FEATURE_NAMES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "reward": asdict(DEFAULT_REWARD),
        "heads": heads,
        "distillation": {
            "file": distilled_path.name,
            "sha256": _sha256(distilled_path),
            "minimum_support": args.min_distill_support,
            "contexts": len(aggregates),
            "context_actions": sum(len(actions) for actions in aggregates.values()),
        },
        "deployment": {
            "native_safety_required": True,
            "unknown_context_fallback": "online-contextual-bandit",
            "offline_outputs_are_outcomes_not_legal-actions": True,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["source"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
