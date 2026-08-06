# Corpus and model compatibility

The durable data boundary is native combat evidence, not a particular policy
or reward formula. A learner may change without invalidating observations.

## Version axes

- `schema_version` describes field names, units, and encoding.
- `reward_version` describes only how outcome fields are scalarized.
- compiled policy artifacts are disposable caches and may be regenerated.

These versions must never be coupled. A schema migration adds fields or derives
them from old fields; it does not rewrite an old scalar reward into a supposed
new observation.

## Storage tiers

1. `runtime/th105_opponent_models.json` is the durable compact corpus. It keeps
   bounded sufficient statistics per opponent/context/action: trials, hits,
   native damage in raw units and basis points, self-damage, spirit use,
   commitment, startup, punish frequency, an eight-bin downside histogram,
   terminal eligibility credit, action timing, attack geometry, projectile
   envelopes, defense outcomes, and observed cancel edges.
2. `runtime/th105_human_demonstrations.json` keeps facing-normalized,
   run-length-encoded human input patterns and their raw outcome components.
   These are cold-start evidence, not permanent rules.
3. `runtime/th105_live.jsonl` is bounded diagnostic replay data. It may be
   rotated and must not be the only source for a learned fact.
4. `runtime/th105_compiled_policy.json` and `runtime/th105_human_policy.json`
   are read-only caches. They can always be rebuilt from tiers 1 and 2.
5. `runtime/th105_transitions.jsonl` is the bounded staging corpus for
   event-aligned semi-Markov transitions. Each row keeps the quantized start
   state, exact native-gated legal action set, chosen
   action and behavior probability, duration, raw outcome components, next
   state, opponent/difficulty, and independent schema versions. Action schema
   v2 uses an exact singleton after a hard priority/commitment gate and a full
   set wherever several candidates survive; legacy unknown sets remain `null`.
   A legal-set change is itself an option boundary. The live JSONL plus eight gzip
   rotations are capped; an offline compactor can later produce immutable
   stratified Parquet shards. The concrete training/deployment contract is in
   [`OFFLINE_TRAINING.md`](OFFLINE_TRAINING.md).

Damage is normalized to basis points of the observed maximum (`10000 == 100%`)
while legacy raw counters remain present. This permits cross-character and
future balance-version comparisons without destroying exact TH105 values.
Histograms preserve a compact approximation of tail risk, which a mean alone
cannot recover.

## Compatibility rules

- Readers accept missing or partially mixed fields and reconstruct every legacy
  trial's v1 normalization from TH105's 10000 HP / 1000 spirit scale before
  adding new evidence. Trial confidence and normalized outcome mass therefore
  cannot diverge after the first v1 sample.
- New writers retain raw outcome components; they never store only Q, reward,
  logits, or an action rank.
- Unknown additive fields are ignored by old readers.
- Model compilers calculate utility from the current reward module and stamp
  its `reward_version`; changing reward weights requires recompilation, not
  recollection.
- Context/action identifiers are opaque stable strings. Character-specific
  meaning belongs to learned native IDs and geometry, not schema conditionals.

## Future offline training

The compact aggregates directly support contextual bandits, Bayesian tables,
decision trees, and reward reweighting. High-capacity offline RL additionally
needs option-level state/action/next-state transitions; aggregate means cannot
reconstruct counterfactual temporal structure. A future trainer exports a
versioned, calibrated lookup/tree/int8 ONNX scorer, while online deployment
continues to enforce native legality and hazard constraints independently.
