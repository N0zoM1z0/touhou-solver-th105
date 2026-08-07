from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.action_selection import CoverageExplorer


class CoverageExplorerTests(unittest.TestCase):
    def test_zero_exploration_is_deterministic_and_reports_probability(self) -> None:
        explorer = CoverageExplorer(seed=1)
        selected = explorer.choose(
            "ctx", {"guard": 3.0, "jump": 2.0}, exploration_rate=0.0
        )
        self.assertEqual(selected.action, "guard")
        self.assertEqual(selected.behavior_probability, 1.0)
        self.assertEqual(selected.legal_actions, ("guard", "jump"))

    def test_full_exploration_moves_mass_to_less_selected_action(self) -> None:
        explorer = CoverageExplorer(seed=2)
        explorer.selected["ctx|guard"] = 99
        selection = explorer.choose(
            "ctx", {"guard": 100.0, "jump": 0.0}, exploration_rate=1.0
        )
        # The chosen draw is incidental; its logged propensity proves that the
        # rare action owns most of the exploration distribution.
        if selection.action == "jump":
            self.assertGreater(selection.behavior_probability, 0.8)
        else:
            self.assertLess(selection.behavior_probability, 0.2)

    def test_checkpoint_preserves_coverage_counts(self) -> None:
        first = CoverageExplorer(seed=3)
        first.choose("ctx", {"a": 1.0, "b": 0.0}, exploration_rate=0.5)
        second = CoverageExplorer(seed=3)
        second.import_state(first.export_state())
        self.assertEqual(second.selected, first.selected)
        self.assertEqual(second.opportunities, first.opportunities)
        self.assertEqual(second.scope_decisions, first.scope_decisions)

    def test_exploration_rate_decays_to_nonzero_floor(self) -> None:
        explorer = CoverageExplorer(seed=4)
        first = explorer.choose("ctx", {"a": 1.0, "b": 0.0}, exploration_rate=0.08)
        for _ in range(400):
            last = explorer.choose(
                "ctx", {"a": 1.0, "b": 0.0}, exploration_rate=0.08
            )
        self.assertLess(last.exploration_rate, first.exploration_rate)
        self.assertGreaterEqual(last.exploration_rate, 0.015)

    def test_equivalent_exact_edges_share_one_coverage_counter(self) -> None:
        explorer = CoverageExplorer(seed=5)
        explorer.choose(
            "cancel:mid:ground:advantage",
            {"edge-a": 1.0, "edge-b": 0.0},
            exploration_rate=0.5,
            coverage_keys={"edge-a": "input:z", "edge-b": "input:z"},
        )
        self.assertEqual(len(explorer.opportunities), 1)
        self.assertEqual(len(explorer.selected), 1)


if __name__ == "__main__":
    unittest.main()
