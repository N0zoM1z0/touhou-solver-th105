# Offline training and deployment contract

This document defines how a multi-core CPU trainer can improve the TH105 agent
without moving expensive or opaque inference into the live game loop. The
native legality, resource, geometry, and finite-horizon hazard checks remain the
authority. A learned model only ranks candidates that those checks permit.

## Durable training unit

Full 60 Hz frame dumps are redundant and grow quickly. The live agent now
stages one event-aligned semi-Markov transition for each intent segment (attack,
defense, movement, or bounded neutral option) in `th105_transitions.jsonl`:

```json
{
  "schema_version": 1,
  "feature_schema_version": 1,
  "action_schema_version": 1,
  "game_build_sha256": "...",
  "episode_id": "...",
  "step": 17,
  "opponent": "0x006B22DC",
  "difficulty": "hard",
  "policy_generation": "...",
  "state": {
    "context": "close:ground:corner:recovery",
    "relative_x_q": 41,
    "relative_y_q": 0,
    "closing_speed_q": -6,
    "self_hp_bp": 7820,
    "enemy_hp_bp": 6140,
    "spirit_bp": 8000,
    "enemy_action": 308,
    "enemy_sequence": 0,
    "enemy_pose": 4,
    "enemy_phase": "recovery",
    "projectile_risk_q": 23
  },
  "legal_actions": ["high_guard", "jump_back", "214B"],
  "action": "214B",
  "behavior_probability": 0.18,
  "duration_frames": 31,
  "outcome": {
    "damage_bp": 930,
    "self_damage_bp": 0,
    "spirit_cost_bp": 2000,
    "punished": false,
    "terminal": null
  },
  "next_state": {"context": "mid:ground:field:neutral"}
}
```

State values are fixed-width integers or bounded categories. Positions and
velocities are quantized; HP and spirit use basis points. The legal-action set
is essential: an offline trainer must not interpret a native-illegal option as
an action the behavior policy deliberately rejected. The behavior probability
is also required for importance-weighted evaluation; deterministic choices use
`1.0`, while exploration records its real sampling probability. The current
policy is deterministic and not every branch exports its exact candidate set,
so those rows deliberately carry `legal_actions: null` and are excluded from
algorithms that require action support. This is honest missingness, not an empty
legal set.

Rows should be dictionary encoded into Arrow/Parquet shards with Zstd
compression. Shards are immutable, checksummed, and listed in a manifest. A
target size of 32--128 MiB avoids millions of tiny files. Storage stays bounded
by retaining every terminal, damage, rare-action, and high-tail-loss transition
while using stratified reservoir sampling for repetitive neutral states. This
preserves unusual failures without storing hours of identical idle frames.

## Outcome learning is separate from gameplay preference

Damage dealt, self-damage, connection probability, spirit cost, punish
probability, commitment time, downside quantiles, and terminal outcome are the
learned targets. They are physical observations; none of them contains the
agent's current offensive, defensive, or risk preference. The corpus, CPU model
heads, and distilled runtime table therefore retain these components separately
instead of replacing them with one scalar reward.

Scalarization happens at action-selection time. The runtime combines predicted
outcomes with a versioned `RewardConfig`, for example:

```text
damage_weight * damage
- self_damage_weight * self_damage
- tail_risk_weight * excess_downside
- resource, whiff, punish, and commitment costs
+ terminal_weight * terminal_value
```

This boundary is a design contract:

- changing gameplay weights re-scores the same online statistics and distilled
  outcome table; it does not require new gameplay collection or outcome-model
  training;
- an A/B test changes one versioned reward profile while keeping the corpus,
  model hashes, native gates, opponent/difficulty strata, and round allocation
  fixed;
- training loss, tree depth, regularization, calibration, and uncertainty are
  model hyperparameters, not gameplay preference weights;
- artifacts must expose outcome heads and support counts. A trainer must not
  publish only a reward-weighted ranking if the underlying outcomes are
  available;
- native safety, resource, geometry, startup, and recovery gates remain outside
  scalarization and cannot be weakened merely by changing a reward profile.

A future fitted-Q or direct policy model is reward-dependent because its target
already includes long-horizon scalar return. Such artifacts must declare their
reward profile and be retrained or emitted once per profile. Prefer vector-valued
Q/outcome heads or successor-feature-like representations when practical so
several gameplay preferences can still share one expensive CPU training run.
Regardless of model family, the raw transition corpus remains reusable.

## CPU-first training progression

The first baseline is deliberately tabular and CPU-native:

1. Reproduce the online contextual-bandit ranking from held-out data. This
   validates feature extraction and reward reconstruction.
2. Train separate categorical histogram-tree heads for damage, self-damage,
   downside quantiles, spirit cost, punish probability, and terminal credit.
   Keep these outcomes separate so reward changes do not require recollection.
3. When sequential action coverage is adequate, compare conservative fitted-Q
   iteration with boosted trees. Penalize actions outside observed support;
   deterministic rows with unknown legal sets cannot justify counterfactual
   off-policy claims.
4. Train several time/opponent folds or bootstrap heads. Their disagreement is
   deployed as uncertainty, not averaged away during evaluation.
5. Distill the best CPU ensemble into a bounded lookup table or compiled tree
   scorer. A larger training model is not automatically a better runtime model.

`scripts/train_th105_cpu.py` implements the multi-head boosted-tree baseline and
distills an algorithm-neutral `distilled_policy.json`. Its categorical action,
opponent, and difficulty features avoid lossy integer IDs. Complete episodes are
held out chronologically, never split into adjacent random rows. Independent
heads can be trained concurrently across a many-core server; each individual
head may use fewer threads if memory bandwidth becomes the bottleneck.

Separate heads prevent abundant guard samples from drowning sparse combo and
skill evidence. The engine's tactical gate selects defense, offense, or neutral;
the corresponding head ranks exact timing/options. A later model may learn the
gate, but only after it can beat the native gate in held-out and live tests.

## Runtime artifact

CPU training produces a self-contained directory, not a replacement corpus:

```text
policy-th105-sakuya-0007/
  manifest.json
  distilled_policy.json
  models/
    damage_bp.cbm
    self_damage_bp.cbm
    self_damage_p90_bp.cbm
    spirit_cost_bp.cbm
    punished_probability.cbm
    terminal_value.cbm
```

`manifest.json` records artifact/schema/reward versions, exact game build,
training-corpus manifest hashes, supported characters/difficulties, model SHA,
feature order, categorical features, per-head hashes/metrics, and required
native-safety ABI. The scorer returns per-action expected outcomes,
downside/tail estimate, and uncertainty. Online
selection uses a risk-adjusted score such as

```text
expected utility - risk_weight * downside - uncertainty_weight * disagreement
```

and falls back to the current table learner when the state is out of
distribution, the artifact is incompatible, inference exceeds its deadline, or
all scored actions fail the native gate. The distilled table is the first
deployment backend; a future compiled native-tree backend must use the same
manifest and output contract and remain CPU-only.

## Session export and Hub layout

`scripts/export_th105_session.py` extracts one controller `session_id` across
the immutable content-addressed corpus archive, bounded current JSONL, and gzip
rotations, deduplicates transition IDs, and emits deterministic gzip files.
`scripts/archive_th105_corpus.py` preserves each compressed transition rotation
without pausing gameplay; future controllers also archive an oldest shard
synchronously before bounded telemetry evicts it. It exports sanitized terminal summaries rather
than raw live telemetry, which deliberately excludes plugin paths and large
frame snapshots. Baseline/final model files, game/policy/schema hashes, reward
weights, action coverage, and privacy assertions are recorded in `manifest.json`.

The portable dataset directory can be uploaded unchanged to a Hugging Face
dataset repository. The repository starts private until the sensitive-fragment
scan, schema validation, and artifact checks pass. Credentials remain in the
user-level Hub credential store and are never part of the export.

## Evaluation protocol

Training loss is not a gameplay metric. Offline checks stay intentionally
lightweight: they catch broken data, unsupported claims, gross regressions, and
runtime incompatibility. They do not veto a candidate merely because one proxy
metric moved in an unexpected direction.

### Data and split integrity

- Split by complete campaign/session and time, never by random adjacent frames.
- Keep opponent and difficulty strata visible; include a held-out future-time
  slice and at least one low-data matchup slice.
- Assert no episode IDs, policy generations, or overlapping option windows cross
  train/validation/test boundaries.
- Report action coverage and effective sample size. Unsupported actions cannot
  receive a trustworthy offline claim.

### Offline policy diagnostics

- Held-out action ranking and top-k regret against observed option outcomes.
- Calibration of predicted damage, self-damage, connection probability, and
  downside quantiles.
- Doubly robust / fitted-Q evaluation with confidence intervals. Weighted
  importance sampling is reported only when effective sample size is adequate.
- Stress slices for low spirit, corners, airborne states, repeated attack chains,
  spells, and high projectile density.

Off-policy estimates are advisory screening evidence, not promotion gates: the
corpus does not contain outcomes for actions never tried, and a complex policy
can trade one proxy metric for a better real sequence of states.

### Shadow and live evaluation

A candidate first runs in shadow mode, scoring the live legal set without
controlling input. This verifies feature parity, OOD rate, inference latency,
and disagreement with the resident policy. It then enters an interleaved A/B
test over complete rounds, stratified by opponent and difficulty. Promotion uses
bootstrap confidence intervals over rounds/campaigns rather than treating
correlated frames as independent samples.

Physical complete-round performance is authoritative. The primary metric is win
rate; normalized HP differential is a useful lower-variance secondary signal
while the win sample is small. The following remain diagnostic views rather
than a mandatory multi-objective scorecard:

- damage taken per enemy active window and repeated-chain defense rate;
- punish-window conversion, connection rate, and damage per committed frame;
- spirit efficiency and low-spirit failure rate;
- 90/95% downside damage and catastrophic-trade rate;
- action/skill diversity, OOD/fallback rate, and native-gate rejection rate;
- inference p50/p95/p99, missed-frame count, crashes, and stuck-input events.

A model is rejected automatically only for hard failures: incompatible schema
or game build, native-safety violation, crash/stuck input, corrupt outputs, or
missed inference deadlines. Otherwise it receives a bounded physical A/B run.
The candidate with better complete-round play is retained; inconclusive results
collect more rounds instead of being discarded by proxy metrics. Its
`evaluation.json` remains beside the artifact so the decision is auditable and
reversible.

## Reusability rule

Raw transition components and corpus manifests are permanent evidence. Scalar
rewards, Q values, rankings, neural weights, and compiled policies are derived
artifacts. A new reward re-scores the current outcome model; it retrains only
reward-dependent artifacts such as scalar Q values or direct policies. A new
algorithm trains from the same transition shards and never rewrites old
observations to resemble the new policy.
