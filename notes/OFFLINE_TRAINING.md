# Offline training and deployment contract

This document defines how a future GPU trainer can improve the TH105 agent
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

## Training progression

The first GPU baseline should be deliberately modest:

1. Reproduce the online contextual-bandit ranking from held-out data. This
   validates feature extraction and reward reconstruction.
2. Train separate offense and defense value heads with a shared small encoder.
   Discrete-action IQL or conservative Q-learning is a suitable starting point;
   behavior-cloning regularization prevents unsupported action extrapolation.
3. Train several seeds or bootstrap heads. Their disagreement is deployed as
   uncertainty, not averaged away during evaluation.
4. Distill the best ensemble into a small int8 MLP, tree ensemble, or lookup
   table. A large training model is not automatically the best runtime model.

Separate heads prevent abundant guard samples from drowning sparse combo and
skill evidence. The engine's tactical gate selects defense, offense, or neutral;
the corresponding head ranks exact timing/options. A later model may learn the
gate, but only after it can beat the native gate in held-out and live tests.

## Runtime artifact

GPU training produces a self-contained directory, not a replacement corpus:

```text
policy-th105-sakuya-0007/
  manifest.json
  scorer.int8.onnx
  feature_schema.json
  action_schema.json
  calibration.json
  evaluation.json
```

`manifest.json` records artifact/schema/reward versions, exact game build,
training-corpus manifest hashes, supported characters/difficulties, model SHA,
input/output tensor names, and required native-safety ABI. The scorer returns
per-action expected utility, downside/tail estimate, and uncertainty. Online
selection uses a risk-adjusted score such as

```text
expected utility - risk_weight * downside - uncertainty_weight * disagreement
```

and falls back to the current table learner when the state is out of
distribution, the artifact is incompatible, inference exceeds its deadline, or
all scored actions fail the native gate. ONNX Runtime is a convenient first
backend; a tree/lookup artifact should use the same manifest and output contract.

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
artifacts. A new reward or algorithm retrains from the same transition shards;
it never rewrites old observations to resemble the new policy.
