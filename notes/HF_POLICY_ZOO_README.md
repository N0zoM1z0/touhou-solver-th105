---
library_name: catboost
tags:
  - touhou
  - th105
  - offline-rl
  - contextual-bandit
  - game-ai
  - tabular-regression
---

# TH105 offline policy zoo

Private CPU-trained outcome-model bundles for
[`touhou-solver-th105`](https://github.com/N0zoM1z0/touhou-solver-th105).

The source transition corpus is stored separately in the private dataset
repository `Joh1rreq/touhou-solver-th105-corpus`. This repository contains
derived model artifacts only; it does not contain the game, credentials, raw
memory dumps, or the private transition corpus.

Each candidate keeps damage, connection probability, self-damage, tail loss,
spirit cost, punish probability, commitment time, and terminal value as
separate outcome heads. Candidates are distilled to the same bounded runtime
lookup contract so they can be compared in shadow mode and complete-round A/B
tests without weakening native legality, geometry, resource, or hazard gates.

## Layout

```text
lunatic30/
  catboost-baseline/
    manifest.json
    distilled_policy.json
    models/
  catboost-ensemble5/
  xgboost-hist/
  extra-trees/
  comparison.json
```

Additional algorithm families use sibling directories under `lunatic30/`.
The artifact manifest records the exact corpus hashes, feature schema, model
hashes, held-out metrics, and runtime compatibility.

Offline metrics are diagnostic. Promotion requires compatible-game shadow
validation followed by complete physical round evaluation.
