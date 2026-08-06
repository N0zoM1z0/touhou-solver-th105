#!/usr/bin/env python3
"""Run reproducible fixed-Lunatic physical comparisons of offline policies."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from install_th105_offline_policy import install_bundle
from th105.session_export import MODEL_FILES, export_session


DEFAULT_CANDIDATES = (
    "extra-trees",
    "xgboost-hist",
    "catboost-ensemble5",
    "catboost-baseline",
)
DEFAULT_WINDOWS_PYTHON = Path(
    "/mnt/c/Users/21992/AppData/Local/Microsoft/WindowsApps/python.exe"
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def copy_models(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def restore_models(baseline: Path, runtime: Path) -> None:
    for name in MODEL_FILES:
        source = baseline / name
        destination = runtime / name
        if source.is_file():
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()


def stop_game() -> None:
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-Process -Name th105c -ErrorAction SilentlyContinue | Stop-Process -Force",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def controller_result(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("controller result is not an object")
    fight = value.get("fight")
    if not isinstance(fight, dict) or not fight.get("session_id"):
        raise ValueError("controller result has no fight session")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install each distilled policy, restore the same online baseline, "
            "and collect complete fixed-Lunatic physical rounds."
        )
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument(
        "--policy-root", type=Path, default=Path("runtime/policy-zoo/lunatic30")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-python", type=Path, default=DEFAULT_WINDOWS_PYTHON)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        choices=DEFAULT_CANDIDATES,
        help="candidate to run; repeat to override the default four-policy order",
    )
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must not already contain files")
    if not args.windows_python.is_file():
        parser.error("Windows Python executable was not found")

    runtime = args.runtime_dir.resolve()
    output = args.output.resolve()
    policy_root = args.policy_root.resolve()
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    baseline = output / "baseline"
    copy_models(runtime, baseline)
    if not (baseline / "th105_opponent_models.json").is_file():
        parser.error("runtime has no online opponent-model baseline")

    experiment: dict[str, object] = {
        "schema_version": 1,
        "difficulty": "lunatic",
        "target_rounds_per_candidate": args.rounds,
        "candidate_order": list(candidates),
        "baseline_files": sorted(path.name for path in baseline.iterdir()),
        "started_at": time.time(),
        "runs": [],
        "status": "running",
    }
    write_json(output / "experiment.json", experiment)

    repository = Path(__file__).resolve().parents[1]
    stop_game()
    try:
        for candidate in candidates:
            candidate_output = output / candidate
            candidate_output.mkdir(parents=True, exist_ok=True)
            restore_models(baseline, runtime)
            installed = install_bundle(policy_root / candidate, runtime)
            started_at = time.time()
            progress = {
                "candidate": candidate,
                "status": "running",
                "started_at": started_at,
                "installed": installed,
            }
            write_json(output / "current.json", progress)

            command = [
                str(args.windows_python),
                "scripts/run_th105_agent.py",
                "auto-arcade",
                "--launch",
                "--p1-character",
                "sakuya",
                "--difficulty",
                "lunatic",
                "--continuous",
                "--round-limit",
                str(args.rounds),
                "--policy",
                "adaptive",
                "--timeout",
                str(args.timeout),
            ]
            stdout_path = candidate_output / "controller.json"
            stderr_path = candidate_output / "controller.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    text=True,
                )
            ended_at = time.time()
            stop_game()
            if completed.returncode != 0:
                progress.update(
                    status="failed", ended_at=ended_at, returncode=completed.returncode
                )
                write_json(output / "current.json", progress)
                raise RuntimeError(
                    f"{candidate} controller exited with {completed.returncode}"
                )

            result = controller_result(stdout_path)
            fight = result["fight"]
            assert isinstance(fight, dict)
            session_id = str(fight["session_id"])
            manifest = export_session(
                runtime_dir=runtime,
                output_dir=candidate_output / "session",
                session_id=session_id,
                started_at=started_at,
                ended_at=ended_at,
                experiment_name=f"lunatic-policy-ab-{candidate}",
                target_rounds=args.rounds,
                baseline_dir=baseline,
            )
            run = {
                "candidate": candidate,
                "status": "complete",
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": ended_at - started_at,
                "session_id": session_id,
                "installed": installed,
                "controller_completed_rounds": fight.get("completed_session_rounds"),
                "statistics": manifest["statistics"],
                "offline_policy_sha256": manifest["offline_policy_sha256"],
            }
            runs = experiment["runs"]
            assert isinstance(runs, list)
            runs.append(run)
            write_json(candidate_output / "result.json", run)
            write_json(output / "experiment.json", experiment)
            write_json(output / "current.json", {**run, "next": "pending"})
    finally:
        stop_game()
        restore_models(baseline, runtime)

    experiment["status"] = "complete"
    experiment["ended_at"] = time.time()
    write_json(output / "experiment.json", experiment)
    (output / "current.json").unlink(missing_ok=True)
    print(json.dumps(experiment, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
