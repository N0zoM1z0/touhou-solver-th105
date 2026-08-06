#!/usr/bin/env python3
"""Train alternative CPU outcome-model families with one distilled contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th105.offline_cpu import (
    CATEGORICAL_FEATURES,
    CPU_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    candidate_prediction_records,
    distillation_context,
    feature_vector,
    outcome_targets,
    temporal_episode_split,
    validate_transition_schemas,
)
from th105.reward import DEFAULT_REWARD, REWARD_VERSION


ARTIFACT_SCHEMA_VERSION = 1
HEAD_SPECS: tuple[tuple[str, str, Callable[[dict[str, float]], float]], ...] = (
    ("damage_bp", "mean", lambda target: target["damage_bp"]),
    (
        "connection_probability",
        "mean",
        lambda target: target["connection_probability"],
    ),
    ("self_damage_bp", "mean", lambda target: target["self_damage_bp"]),
    (
        "self_damage_p90_bp",
        "quantile-0.9",
        lambda target: target["self_damage_bp"],
    ),
    ("spirit_cost_bp", "mean", lambda target: target["spirit_cost_bp"]),
    (
        "punished_probability",
        "mean",
        lambda target: target["punished_probability"],
    ),
    (
        "commitment_frames",
        "mean",
        lambda target: target["commitment_frames"],
    ),
    ("terminal_value", "mean", lambda target: target["terminal_value"]),
)


def _load_records(path: Path) -> list[dict[str, object]]:
    import gzip

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


def _feature_frame(rows: list[list[str | float]]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for policy-zoo training; install "
            "requirements-cpu-train-zoo.txt"
        ) from exc

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].astype(str).astype("category")
    for name in NUMERIC_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return frame


def _constant_head(
    value: float, validation_values: list[float]
) -> tuple[list[float], dict[str, object]]:
    return [value], {
        "kind": "constant",
        "value": value,
        "validation": _metrics(validation_values, [value] * len(validation_values)),
    }


def _train_catboost_ensemble(
    *,
    rows: list[list[str | float]],
    targets: list[dict[str, float]],
    train_indices: list[int],
    validation_indices: list[int],
    models_dir: Path,
    threads: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    members: int,
) -> tuple[dict[str, list[float]], dict[str, dict[str, object]], dict[str, str]]:
    try:
        import catboost
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise SystemExit(
            "CatBoost is required; install requirements-cpu-train-zoo.txt"
        ) from exc

    import numpy as np

    categorical_indices = [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
    train_rows = [rows[index] for index in train_indices]
    validation_rows = [rows[index] for index in validation_indices]
    predictions: dict[str, list[float]] = {}
    heads: dict[str, dict[str, object]] = {}
    for head_index, (name, mode, target_fn) in enumerate(HEAD_SPECS):
        values = [float(target_fn(target)) for target in targets]
        train_values = [values[index] for index in train_indices]
        validation_values = [values[index] for index in validation_indices]
        if max(train_values) == min(train_values):
            constant = train_values[0]
            predictions[name] = [constant] * len(rows)
            _, heads[name] = _constant_head(constant, validation_values)
            continue
        member_dir = models_dir / name
        member_dir.mkdir(exist_ok=True)
        member_predictions = []
        member_metadata: list[dict[str, object]] = []
        loss = "Quantile:alpha=0.9" if mode == "quantile-0.9" else "RMSE"
        for member in range(members):
            seed = 20500 + head_index * 100 + member
            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                loss_function=loss,
                random_seed=seed,
                thread_count=threads,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(
                train_rows,
                train_values,
                cat_features=categorical_indices,
                eval_set=(validation_rows, validation_values),
                early_stopping_rounds=max(20, iterations // 10),
                verbose=False,
            )
            predicted = np.asarray(model.predict(rows), dtype=float)
            member_predictions.append(predicted)
            model_path = member_dir / f"seed-{seed}.cbm"
            model.save_model(model_path)
            member_metadata.append(
                {
                    "seed": seed,
                    "file": model_path.relative_to(models_dir.parent).as_posix(),
                    "sha256": _sha256(model_path),
                    "best_iteration": int(model.get_best_iteration()),
                    "validation": _metrics(
                        validation_values,
                        [float(predicted[index]) for index in validation_indices],
                    ),
                }
            )
        averaged = np.mean(np.stack(member_predictions), axis=0)
        predictions[name] = [float(value) for value in averaged]
        heads[name] = {
            "kind": "catboost-regressor-ensemble",
            "aggregation": "mean",
            "loss": loss,
            "members": member_metadata,
            "validation": _metrics(
                validation_values,
                [float(averaged[index]) for index in validation_indices],
            ),
        }
    return predictions, heads, {"catboost": catboost.__version__}


def _train_xgboost(
    *,
    rows: list[list[str | float]],
    targets: list[dict[str, float]],
    train_indices: list[int],
    validation_indices: list[int],
    models_dir: Path,
    threads: int,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> tuple[dict[str, list[float]], dict[str, dict[str, object]], dict[str, str]]:
    try:
        import xgboost
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit(
            "XGBoost is required; install requirements-cpu-train-zoo.txt"
        ) from exc

    frame = _feature_frame(rows)
    train_frame = frame.iloc[train_indices]
    validation_frame = frame.iloc[validation_indices]
    predictions: dict[str, list[float]] = {}
    heads: dict[str, dict[str, object]] = {}
    for head_index, (name, mode, target_fn) in enumerate(HEAD_SPECS):
        values = [float(target_fn(target)) for target in targets]
        train_values = [values[index] for index in train_indices]
        validation_values = [values[index] for index in validation_indices]
        if max(train_values) == min(train_values):
            constant = train_values[0]
            predictions[name] = [constant] * len(rows)
            _, heads[name] = _constant_head(constant, validation_values)
            continue
        quantile = mode == "quantile-0.9"
        parameters: dict[str, object] = {
            "n_estimators": iterations,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3.0,
            "reg_lambda": 5.0,
            "reg_alpha": 0.05,
            "tree_method": "hist",
            "enable_categorical": True,
            "n_jobs": threads,
            "random_state": 30500 + head_index,
            "early_stopping_rounds": max(20, iterations // 10),
            "verbosity": 0,
        }
        if quantile:
            parameters.update(
                {
                    "objective": "reg:quantileerror",
                    "quantile_alpha": 0.9,
                    "eval_metric": "quantile",
                }
            )
        else:
            parameters.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
        model = XGBRegressor(**parameters)
        model.fit(
            train_frame,
            train_values,
            eval_set=[(validation_frame, validation_values)],
            verbose=False,
        )
        predicted = [float(value) for value in model.predict(frame)]
        model_path = models_dir / f"{name}.ubj"
        model.save_model(model_path)
        predictions[name] = predicted
        heads[name] = {
            "kind": "xgboost-regressor",
            "objective": parameters["objective"],
            "file": model_path.relative_to(models_dir.parent).as_posix(),
            "sha256": _sha256(model_path),
            "best_iteration": int(model.best_iteration),
            "validation": _metrics(
                validation_values,
                [predicted[index] for index in validation_indices],
            ),
        }
    return predictions, heads, {"xgboost": xgboost.__version__}


def _extra_trees_predictions(model: object, rows: object, *, quantile: bool):
    if not quantile:
        return model.predict(rows)
    import numpy as np

    estimators = getattr(model, "estimators_")
    values = np.stack([tree.predict(rows) for tree in estimators])
    return np.quantile(values, 0.9, axis=0)


def _train_extra_trees(
    *,
    rows: list[list[str | float]],
    targets: list[dict[str, float]],
    train_indices: list[int],
    validation_indices: list[int],
    models_dir: Path,
    threads: int,
    trees: int,
    depth: int,
) -> tuple[dict[str, list[float]], dict[str, dict[str, object]], dict[str, str]]:
    try:
        import joblib
        import numpy as np
        import sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required; install requirements-cpu-train-zoo.txt"
        ) from exc

    frame = _feature_frame(rows)
    preprocessor = ColumnTransformer(
        (
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float32
                ),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ),
        remainder="drop",
    )
    encoded_train = preprocessor.fit_transform(frame.iloc[train_indices])
    encoded_all = preprocessor.transform(frame)
    encoder_path = models_dir / "feature_encoder.joblib"
    joblib.dump(preprocessor, encoder_path, compress=3)
    predictions: dict[str, list[float]] = {}
    heads: dict[str, dict[str, object]] = {}
    for head_index, (name, mode, target_fn) in enumerate(HEAD_SPECS):
        values = [float(target_fn(target)) for target in targets]
        train_values = [values[index] for index in train_indices]
        validation_values = [values[index] for index in validation_indices]
        if max(train_values) == min(train_values):
            constant = train_values[0]
            predictions[name] = [constant] * len(rows)
            _, heads[name] = _constant_head(constant, validation_values)
            continue
        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=threads,
            random_state=40500 + head_index,
        )
        model.fit(encoded_train, train_values)
        predicted_array = _extra_trees_predictions(
            model, encoded_all, quantile=mode == "quantile-0.9"
        )
        predicted = [float(value) for value in predicted_array]
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        predictions[name] = predicted
        heads[name] = {
            "kind": "extra-trees-regressor",
            "aggregation": "tree-quantile-0.9" if mode == "quantile-0.9" else "mean",
            "file": model_path.relative_to(models_dir.parent).as_posix(),
            "sha256": _sha256(model_path),
            "validation": _metrics(
                validation_values,
                [predicted[index] for index in validation_indices],
            ),
        }
    return (
        predictions,
        heads,
        {
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "feature_encoder_file": encoder_path.relative_to(
                models_dir.parent
            ).as_posix(),
            "feature_encoder_sha256": _sha256(encoder_path),
        },
    )


def _distilled_payload(
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    *,
    corpus_manifest: dict[str, object],
    min_support: int,
) -> dict[str, object]:
    aggregates: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    working: dict[tuple[str, str], dict[str, object]] = {}
    for row_index, record in enumerate(records):
        context = distillation_context(record)
        action = str(record.get("action", "unknown"))
        entry = working.setdefault(
            (context, action),
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
        if support < min_support:
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
    statistics = corpus_manifest.get("statistics", {})
    if not isinstance(statistics, dict):
        statistics = {}
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "context_schema_version": 1,
        "outcome_heads": sorted(predictions),
        "compatibility": {
            "game_build_sha256": corpus_manifest.get("game_build_sha256", []),
            "difficulties": statistics.get("difficulties", []),
            "opponents": statistics.get("opponents", []),
        },
        "contexts": {
            context: actions for context, actions in sorted(aggregates.items())
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("catboost-ensemble", "xgboost", "extra-trees"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--trees", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-distill-support", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.threads <= 0
        or args.iterations <= 0
        or args.trees <= 0
        or args.depth <= 0
        or args.members <= 0
        or args.min_distill_support <= 0
    ):
        raise SystemExit("thread, model-size, and support arguments must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    records = _load_records(args.input)
    validate_transition_schemas(records)
    corpus_manifest: dict[str, object] = {}
    if args.corpus_manifest and args.corpus_manifest.is_file():
        value = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("corpus manifest must be a JSON object")
        corpus_manifest = value
    prediction_records = candidate_prediction_records(records)
    rows = [feature_vector(record) for record in prediction_records]
    targets = [outcome_targets(record) for record in records]
    train_indices, validation_indices = temporal_episode_split(
        records, validation_fraction=args.validation_fraction
    )
    args.output.mkdir(parents=True, exist_ok=True)
    models_dir = args.output / "models"
    models_dir.mkdir(exist_ok=True)

    common = {
        "rows": rows,
        "targets": targets,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "models_dir": models_dir,
        "threads": args.threads,
        "depth": args.depth,
    }
    if args.algorithm == "catboost-ensemble":
        predictions, heads, libraries = _train_catboost_ensemble(
            **common,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            members=args.members,
        )
        parameters = {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "members": args.members,
        }
    elif args.algorithm == "xgboost":
        predictions, heads, libraries = _train_xgboost(
            **common,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
        parameters = {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }
    else:
        predictions, heads, libraries = _train_extra_trees(**common, trees=args.trees)
        parameters = {
            "trees": args.trees,
            "depth": args.depth,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        }

    distilled = _distilled_payload(
        prediction_records,
        predictions,
        corpus_manifest=corpus_manifest,
        min_support=args.min_distill_support,
    )
    distilled_path = args.output / "distilled_policy.json"
    distilled_path.write_text(
        json.dumps(distilled, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    contexts = distilled["contexts"]
    assert isinstance(contexts, dict)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": f"{args.algorithm}-cpu-multihead-with-distillation",
        "trainer": {
            "algorithm": args.algorithm,
            "task_type": "CPU",
            "threads": args.threads,
            "parameters": parameters,
            "libraries": libraries,
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
            "split": "chronological-complete-episode",
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
            "contexts": len(contexts),
            "context_actions": sum(
                len(actions)
                for actions in contexts.values()
                if isinstance(actions, dict)
            ),
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
    print(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "records": len(records),
                "train_records": len(train_indices),
                "validation_records": len(validation_indices),
                "contexts": len(contexts),
                "context_actions": manifest["distillation"]["context_actions"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
