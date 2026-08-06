from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.session_export import export_session

from run_th105_policy_ab import controller_result


def _transition(session: str, transition_id: str, step: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "feature_schema_version": 1,
        "action_schema_version": 1,
        "session_id": session,
        "episode_id": "episode-a",
        "transition_id": transition_id,
        "step": step,
        "opponent": "0x00000001@lunatic",
        "difficulty": "lunatic",
        "game_build_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "offline_policy_sha256": "d" * 64,
        "state": {},
        "legal_actions": None,
        "legal_actions_known": False,
        "action": "guard",
        "behavior_probability": 1.0,
        "outcome": {"punished": False},
        "next_state": {},
    }


class SessionExportTests(unittest.TestCase):
    def test_controller_result_accepts_windows_gbk_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller.json"
            path.write_bytes(
                json.dumps(
                    {"identity": {"path": "东方绯想天"}, "fight": {"session_id": "s"}},
                    ensure_ascii=False,
                ).encode("gbk")
            )
            self.assertEqual(controller_result(path)["fight"]["session_id"], "s")

    def test_exports_only_requested_session_with_sanitized_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            archive = runtime / "corpus_archive"
            archive.mkdir()
            session = "session-a"
            with gzip.open(
                archive / ("th105_transitions.jsonl." + "d" * 64 + ".gz"),
                "wt",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(_transition("other", "other:0", 0)) + "\n")
                handle.write(json.dumps(_transition(session, "episode-a:0", 0)) + "\n")
            (runtime / "th105_transitions.jsonl").write_text(
                json.dumps(_transition(session, "episode-a:1", 1)) + "\n",
                encoding="utf-8",
            )
            live = [
                {
                    "time": 10.0,
                    "event": "encounter-start",
                    "session_id": session,
                    "difficulty": "lunatic",
                    "opponent": "0x00000001@lunatic",
                    "plugin": {"path": "/private/path/must-not-export"},
                },
                {
                    "time": 11.0,
                    "event": "round-terminal",
                    "terminal_sequence": 1,
                    "difficulty": "lunatic",
                    "opponent": "0x00000001@lunatic",
                    "won": True,
                    "me_hp": 5000,
                    "enemy_hp": 0,
                    "max_me_hp": 10000,
                    "max_enemy_hp": 10000,
                },
            ]
            (runtime / "th105_live.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in live), encoding="utf-8"
            )
            for name in (
                "th105_opponent_models.json",
                "th105_compiled_policy.json",
            ):
                (runtime / name).write_text("{}\n", encoding="utf-8")
            output = root / "export"
            manifest = export_session(
                runtime_dir=runtime,
                output_dir=output,
                session_id=session,
                started_at=10.0,
                ended_at=12.0,
                experiment_name="test",
                target_rounds=1,
                source_commit="c" * 40,
            )
            self.assertEqual(manifest["statistics"]["transitions"], 2)
            self.assertEqual(manifest["statistics"]["terminal_rounds"], 1)
            self.assertEqual(manifest["offline_policy_sha256"], ["d" * 64])
            with gzip.open(
                output / "data" / "transitions.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                exported = [json.loads(line) for line in handle]
            self.assertEqual(
                [row["transition_id"] for row in exported],
                ["episode-a:0", "episode-a:1"],
            )
            terminal_text = gzip.open(
                output / "data" / "terminals.jsonl.gz", "rt", encoding="utf-8"
            ).read()
            self.assertNotIn("private", terminal_text)
            self.assertTrue((output / "README.md").is_file())

    def test_rejects_sensitive_transition_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            runtime.mkdir()
            record = _transition("session-a", "episode-a:0", 0)
            record["action"] = "hf_not-a-real-token"
            (runtime / "th105_transitions.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            (runtime / "th105_live.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                export_session(
                    runtime_dir=runtime,
                    output_dir=Path(temporary) / "export",
                    session_id="session-a",
                    started_at=1.0,
                    ended_at=2.0,
                    experiment_name="test",
                    target_rounds=1,
                )


if __name__ == "__main__":
    unittest.main()
