from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from install_th105_offline_policy import install_bundle


class OfflineInstallerTests(unittest.TestCase):
    def test_installs_only_hash_and_build_compatible_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            policy = bundle / "distilled_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "artifact_schema_version": 1,
                        "compatibility": {"game_build_sha256": ["build-a"]},
                        "contexts": {"context": {}},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(policy.read_bytes()).hexdigest()
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_schema_version": 1,
                        "distillation": {
                            "file": "distilled_policy.json",
                            "sha256": digest,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "runtime"
            result = install_bundle(
                bundle, runtime, expected_game_build="build-a"
            )
            self.assertEqual(result["contexts"], 1)
            self.assertEqual(
                hashlib.sha256(
                    (runtime / "th105_offline_policy.json").read_bytes()
                ).hexdigest(),
                digest,
            )

    def test_rejects_hash_mismatch_without_installing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "distilled_policy.json").write_text(
                '{"artifact_schema_version":1,"contexts":{}}\n', encoding="utf-8"
            )
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_schema_version": 1,
                        "distillation": {
                            "file": "distilled_policy.json",
                            "sha256": "bad",
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "runtime"
            with self.assertRaises(ValueError):
                install_bundle(bundle, runtime, expected_game_build="build-a")
            self.assertFalse((runtime / "th105_offline_policy.json").exists())


if __name__ == "__main__":
    unittest.main()
