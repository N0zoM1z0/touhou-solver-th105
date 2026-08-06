---
pretty_name: TH105 Autonomous Grammar Learning Corpus
license: other
tags:
- touhou
- th105
- game-ai
- contextual-bandit
- offline-rl
---

# TH105 autonomous grammar learning corpus

This private dataset is the clean `autonomous-grammar-v4` generation produced
by [touhou-solver-th105](https://github.com/N0zoM1z0/touhou-solver-th105).

The generation starts with zero combat, human-demonstration, fixed-combo,
offline-policy, and cancel-graph evidence. The online agent receives only a
generic input grammar, native game state, legality/resource/hazard gates, and
raw physical outcomes. Character routes and cancel chains must be discovered
from play.

Schema axes are independent:

- transition schema: 2
- feature schema: 1
- action schema: 4
- compact corpus schema: 2
- learning-curve schema: 1

Action schema 4 contains generated directional attack chords and the complete
bounded motion grammar without character-specific action IDs, tactical roles,
damage values, or combo seeds. It is intentionally incompatible with earlier
fixed-route and character-seeded action spaces. Offline trainers reject mixed
action schemas and mixed training generations.

## Layout

- `characters/<p1>/<generation>/generation.json`: immutable generation identity.
- `characters/<p1>/<generation>/baseline/`: empty compact model state.
- `characters/<p1>/<generation>/checkpoints/`: versioned transitions, terminals,
  learning curves, and compact-model snapshots.
- Every session transition also carries the native P1 vtable, so folder identity
  can be validated rather than trusted from its path alone.
- Future character generations use sibling prefixes and never share online
  checkpoints.

The dataset contains no executable, game assets, screenshots, credentials, or
raw process-memory dumps. Raw outcome components remain separate from reward
weights so later CPU training can change scalarization without recollection.
