from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.offense_learning import ActionOutcomeStats
from th105.reward import damage_bin_index, empirical_action_value, upper_tail_mean_bp


def stats_for(outcomes: list[tuple[int, int, int]]) -> ActionOutcomeStats:
    stats = ActionOutcomeStats()
    for damage_bp, self_damage_bp, spirit_bp in outcomes:
        stats.trials += 1
        stats.normalized_samples += 1
        stats.total_damage_bp += damage_bp
        stats.total_self_damage_bp += self_damage_bp
        stats.total_spirit_cost_bp += spirit_bp
        stats.total_commitment += 30
        stats.self_damage_histogram[damage_bin_index(self_damage_bp)] += 1
        if damage_bp:
            stats.connections += 1
        if self_damage_bp:
            stats.punished_trials += 1
    return stats


class RewardTests(unittest.TestCase):
    def test_safe_damage_beats_larger_losing_trade(self) -> None:
        safe = stats_for([(300, 0, 0)] * 5)
        trade = stats_for([(700, 600, 0)] * 5)
        self.assertGreater(empirical_action_value(safe), empirical_action_value(trade))

    def test_whiff_has_real_opportunity_cost(self) -> None:
        whiff = stats_for([(0, 0, 0)] * 3)
        self.assertLess(empirical_action_value(whiff), 0.0)

    def test_tail_risk_penalizes_same_mean_with_bad_outlier(self) -> None:
        steady = stats_for([(500, 250, 0)] * 5)
        risky = stats_for([(500, 0, 0)] * 4 + [(500, 1250, 0)])
        self.assertGreater(empirical_action_value(steady), empirical_action_value(risky))

    def test_histogram_tail_is_bounded_and_compact(self) -> None:
        histogram = [4, 0, 0, 0, 1, 0, 0, 0]
        self.assertEqual(upper_tail_mean_bp(histogram), 1000.0)


if __name__ == "__main__":
    unittest.main()
