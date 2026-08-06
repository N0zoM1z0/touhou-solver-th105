---
pretty_name: TH105 Autonomous Route Learning Corpus
license: other
tags:
- touhou
- th105
- game-ai
- contextual-bandit
- offline-rl
---

# TH105 autonomous route learning corpus

This private dataset is the clean `autonomous-routes-v3` generation produced
by [touhou-solver-th105](https://github.com/N0zoM1z0/touhou-solver-th105).

The generation starts with zero combat, human-demonstration, fixed-combo,
offline-policy, and cancel-graph evidence. The online agent receives only a
generic input grammar, native game state, legality/resource/hazard gates, and
raw physical outcomes. Character routes and cancel chains must be discovered
from play.

Schema axes are independent:

- transition schema: 2
- feature schema: 1
- action schema: 3
- compact corpus schema: 2
- learning-curve schema: 1

Action schema 3 contains generated directional attack chords and motion
hypotheses. It is intentionally incompatible with the earlier fixed-route
action space. Offline trainers reject mixed action schemas and mixed training
generations.

## Layout

- `generation.json`: immutable generation identity and zero-data baseline.
- `baseline/th105_opponent_models.json`: empty compact model state.
- future `checkpoints/`: versioned option-transition, terminal, learning-curve,
  and compact-model snapshots from this generation only.

The dataset contains no executable, game assets, screenshots, credentials, or
raw process-memory dumps. Raw outcome components remain separate from reward
weights so later CPU training can change scalarization without recollection.

