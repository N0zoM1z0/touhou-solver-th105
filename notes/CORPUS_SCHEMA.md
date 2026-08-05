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

Damage is normalized to basis points of the observed maximum (`10000 == 100%`)
while legacy raw counters remain present. This permits cross-character and
future balance-version comparisons without destroying exact TH105 values.
Histograms preserve a compact approximation of tail risk, which a mean alone
cannot recover.

## Compatibility rules

- Readers accept missing fields and reconstruct v1 normalization from TH105's
  legacy 10000 HP / 1000 spirit scale.
- New writers retain raw outcome components; they never store only Q, reward,
  logits, or an action rank.
- Unknown additive fields are ignored by old readers.
- Model compilers calculate utility from the current reward module and stamp
  its `reward_version`; changing reward weights requires recompilation, not
  recollection.
- Context/action identifiers are opaque stable strings. Character-specific
  meaning belongs to learned native IDs and geometry, not schema conditionals.

## Future offline training

The compact corpus directly supports contextual bandits, Bayesian tables,
decision trees, and reward reweighting. A future high-capacity trainer can join
it with bounded diagnostic or option-level replay samples, train on another
machine, then export a small lookup, tree, or ONNX policy. Online deployment
must continue to enforce the native legality and hazard constraints independently
of that model.
