from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.policy_api import PolicyObservation
from th105.policy_loader import HotReloadPolicy


PLUGIN = """
from th105.policy_api import POLICY_API_VERSION, PolicyDecision
class Policy:
    api_version = POLICY_API_VERSION
    name = {name!r}
    def decide(self, observation):
        {body}
    def metrics(self):
        return {{"name": self.name}}
def create_policy():
    return Policy()
"""


class HotReloadPolicyTests(unittest.TestCase):
    def test_reload_and_compile_failure_keep_last_good(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.py"
            path.write_text(
                PLUGIN.format(
                    name="one", body="return PolicyDecision(frozenset({'z'}), 'one')"
                ),
                encoding="utf-8",
            )
            loader = HotReloadPolicy(path, check_interval_frames=1)
            observation = PolicyObservation(1, None, None, ())
            self.assertEqual(loader.decide(observation).intent, "one")

            path.write_text(
                PLUGIN.format(
                    name="two", body="return PolicyDecision(frozenset({'x'}), 'two')"
                ),
                encoding="utf-8",
            )
            self.assertEqual(loader.decide(PolicyObservation(2, None, None, ())).intent, "two")

            path.write_text("this is not python !", encoding="utf-8")
            self.assertEqual(loader.decide(PolicyObservation(3, None, None, ())).intent, "two")
            self.assertEqual(loader.reload_failures, 1)

    def test_runtime_failure_rolls_back_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.py"
            path.write_text(
                PLUGIN.format(
                    name="safe", body="return PolicyDecision(frozenset(), 'safe')"
                ),
                encoding="utf-8",
            )
            loader = HotReloadPolicy(path, check_interval_frames=1)
            path.write_text(
                PLUGIN.format(name="broken", body="raise RuntimeError('boom')"),
                encoding="utf-8",
            )
            decision = loader.decide(PolicyObservation(1, None, None, ()))
            self.assertEqual(decision.intent, "safe")
            self.assertEqual(loader.status()["name"], "safe")

    def test_delayed_runtime_failure_still_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.py"
            path.write_text(
                PLUGIN.format(
                    name="safe", body="return PolicyDecision(frozenset(), 'safe')"
                ),
                encoding="utf-8",
            )
            loader = HotReloadPolicy(path, check_interval_frames=1)
            path.write_text(
                """
from th105.policy_api import POLICY_API_VERSION, PolicyDecision
class Policy:
    api_version = POLICY_API_VERSION
    name = 'delayed-broken'
    def __init__(self): self.calls = 0
    def decide(self, observation):
        self.calls += 1
        if self.calls >= 3: raise RuntimeError('late boom')
        return PolicyDecision(frozenset(), 'new')
    def metrics(self): return {}
def create_policy(): return Policy()
""",
                encoding="utf-8",
            )
            self.assertEqual(loader.decide(PolicyObservation(1, None, None, ())).intent, "new")
            self.assertEqual(loader.decide(PolicyObservation(2, None, None, ())).intent, "new")
            self.assertEqual(loader.decide(PolicyObservation(3, None, None, ())).intent, "safe")
            self.assertEqual(loader.status()["name"], "safe")

    def test_initial_delayed_failure_returns_fail_safe_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.py"
            path.write_text(
                PLUGIN.format(name="broken", body="raise RuntimeError('boom')"),
                encoding="utf-8",
            )
            loader = HotReloadPolicy(path, check_interval_frames=1)
            decision = loader.decide(PolicyObservation(1, None, None, ()))
            self.assertEqual(decision.intent, "policy-error-guard")


if __name__ == "__main__":
    unittest.main()
