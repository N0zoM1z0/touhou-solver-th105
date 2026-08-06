from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.agent import build_parser


class AgentParserTests(unittest.TestCase):
    def test_auto_arcade_is_strict_background_by_default(self) -> None:
        args = build_parser().parse_args(["auto-arcade"])
        self.assertFalse(args.foreground_only)

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

    def test_fixed_difficulty_accepts_a_positive_round_limit(self) -> None:
        args = build_parser().parse_args(
            [
                "auto-arcade",
                "--difficulty",
                "lunatic",
                "--continuous",
                "--round-limit",
                "30",
            ]
        )
        self.assertEqual(args.round_limit, 30)

    def test_round_limit_rejects_zero(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["auto-arcade", "--round-limit", "0"])


if __name__ == "__main__":
    unittest.main()
