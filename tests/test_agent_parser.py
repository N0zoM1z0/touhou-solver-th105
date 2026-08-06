from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.agent import build_parser


class AgentParserTests(unittest.TestCase):
    def test_cycle_uses_weighted_default_quotas(self) -> None:
        args = build_parser().parse_args(
            ["auto-arcade", "--difficulty", "cycle", "--continuous"]
        )
        self.assertEqual(args.cycle_round_quotas, (2, 4, 8, 10))
        self.assertIsNone(args.rounds_per_difficulty)

    def test_equal_quota_override_remains_available(self) -> None:
        args = build_parser().parse_args(
            [
                "auto-arcade",
                "--difficulty",
                "cycle",
                "--rounds-per-difficulty",
                "5",
            ]
        )
        self.assertEqual(args.rounds_per_difficulty, 5)

    def test_custom_weighted_quotas_parse_in_difficulty_order(self) -> None:
        args = build_parser().parse_args(
            [
                "auto-arcade",
                "--difficulty",
                "cycle",
                "--cycle-round-quotas",
                "1,3,7,11",
            ]
        )
        self.assertEqual(args.cycle_round_quotas, (1, 3, 7, 11))


if __name__ == "__main__":
    unittest.main()
