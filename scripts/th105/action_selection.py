"""Bounded coverage exploration over native-gated policy candidates."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSelection:
    action: str
    legal_actions: tuple[str, ...]
    behavior_probability: float
    greedy_action: str
    scores: dict[str, float]


class CoverageExplorer:
    """Epsilon exploration biased toward under-sampled legal actions.

    Native code/policy gates remain authoritative: this class can only choose
    from the candidate mapping supplied by its caller.  Its state is bounded by
    ``scope x action`` sufficient statistics and is safe to checkpoint.
    """

    def __init__(self, seed: int = 105) -> None:
        self.seed = int(seed)
        self.random = random.Random(self.seed)
        self.selected: Counter[str] = Counter()
        self.opportunities: Counter[str] = Counter()
        self.decisions = 0
        self.exploratory_decisions = 0

    @staticmethod
    def _key(scope: str, action: str) -> str:
        return f"{scope}|{action}"

    def choose(
        self,
        scope: str,
        scores: dict[str, float],
        *,
        exploration_rate: float,
    ) -> ActionSelection:
        if not scores:
            raise ValueError("at least one legal action is required")
        legal = tuple(sorted({str(action) for action in scores}))
        bounded_rate = max(0.0, min(1.0, float(exploration_rate)))
        normalized_scores = {action: float(scores[action]) for action in legal}
        greedy = max(legal, key=lambda action: (normalized_scores[action], action))
        for action in legal:
            self.opportunities[self._key(scope, action)] += 1

        if len(legal) == 1 or bounded_rate <= 0.0:
            probabilities = {action: 0.0 for action in legal}
            probabilities[greedy] = 1.0
        else:
            # Inverse-square-root visitation gives rare legal actions more of
            # the exploration mass without allowing a one-off action to
            # dominate forever.
            weights = {
                action: 1.0 / math.sqrt(1.0 + self.selected[self._key(scope, action)])
                for action in legal
            }
            total = sum(weights.values())
            probabilities = {
                action: bounded_rate * weights[action] / total for action in legal
            }
            probabilities[greedy] += 1.0 - bounded_rate

        draw = self.random.random()
        cumulative = 0.0
        chosen = legal[-1]
        for action in legal:
            cumulative += probabilities[action]
            if draw <= cumulative:
                chosen = action
                break
        self.selected[self._key(scope, chosen)] += 1
        self.decisions += 1
        if chosen != greedy:
            self.exploratory_decisions += 1
        return ActionSelection(
            action=chosen,
            legal_actions=legal,
            behavior_probability=max(1e-12, probabilities[chosen]),
            greedy_action=greedy,
            scores=normalized_scores,
        )

    def export_state(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "selected": dict(self.selected),
            "opportunities": dict(self.opportunities),
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
        }

    def import_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        selected = state.get("selected", {})
        opportunities = state.get("opportunities", {})
        if isinstance(selected, dict):
            self.selected.update(
                {str(key): max(0, int(value)) for key, value in selected.items()}
            )
        if isinstance(opportunities, dict):
            self.opportunities.update(
                {str(key): max(0, int(value)) for key, value in opportunities.items()}
            )
        self.decisions = max(0, int(state.get("decisions", self.decisions)))
        self.exploratory_decisions = max(
            0, int(state.get("exploratory_decisions", self.exploratory_decisions))
        )

    def metrics(self) -> dict[str, object]:
        covered = sum(value > 0 for value in self.selected.values())
        available = sum(value > 0 for value in self.opportunities.values())
        return {
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
            "exploratory_rate": (
                self.exploratory_decisions / self.decisions if self.decisions else 0.0
            ),
            "covered_scope_actions": covered,
            "available_scope_actions": available,
            "coverage": covered / available if available else 0.0,
            "selected": dict(self.selected),
            "opportunities": dict(self.opportunities),
        }
