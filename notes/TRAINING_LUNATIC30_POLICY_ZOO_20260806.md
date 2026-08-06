# Lunatic30 CPU policy zoo -- 2026-08-06

## Storage and provenance

Training data, source code, and derived model artifacts are deliberately kept
in separate repositories:

- Private dataset: `Joh1rreq/touhou-solver-th105-corpus` at revision
  `fbd13cb9c900c81d3623a9c7ac515896e2f00800`.
- Training code: Git commit
  `03b8a9e631cff0b5c4eaa5059f01ee09cc88336f`.
- Private model repository: `Joh1rreq/touhou-solver-th105-policy-zoo` at
  revision `ccc35480bfe2c73437b5a9c114c4911dcd80101d`.

The final model revision contains 74 files under `lunatic30/`, totaling
192,375,872 bytes. It was downloaded at the fixed revision and all four bundles
plus the comparison report matched their local copies byte-for-byte. The model
repository was confirmed private.

## Fair comparison contract

All candidates use the same 15,039 transitions, feature schema, reward
components, chronological complete-episode split, and distillation support
threshold. The split contains 12,318 training transitions from the first 24
episodes and 2,721 held-out transitions from the last 6 episodes. Every
candidate emits the same eight outcome heads and a runtime-compatible table
with 946 contexts and 1,234 context-actions.

Three formal candidates ran concurrently with 32 threads each, using all 96
physical CPU cores without oversubscribing them:

| Candidate | Parameters | Wall time | Bundle size |
| --- | --- | ---: | ---: |
| CatBoost baseline | 600 rounds, depth 8, one seed | 29.995 s | 8 MiB |
| CatBoost ensemble | 900 rounds, depth 8, five seeds | 108.563 s | 45 MiB |
| XGBoost histogram | 1,200 rounds, depth 8 | 14.211 s | 25 MiB |
| ExtraTrees | 512 trees per head, depth 18 | 18.998 s | 107 MiB |

The policy-zoo environment used CatBoost 1.2.10, XGBoost 3.4.0,
scikit-learn 1.9.0, and CPython 3.14.6.

## Held-out results

Transition-level mean absolute error on the common held-out episodes:

| Outcome head | Baseline | CB ensemble | XGBoost | ExtraTrees | Lowest |
| --- | ---: | ---: | ---: | ---: | --- |
| damage_bp | 17.4334 | 15.3409 | 15.7205 | 13.7273 | ExtraTrees |
| connection_probability | 0.04410 | 0.04203 | 0.04423 | 0.03836 | ExtraTrees |
| self_damage_bp | 40.2131 | 40.7246 | 34.6729 | 24.6134 | ExtraTrees |
| self_damage_p90_bp | 42.8672 | 41.2944 | 24.2598 | 33.6190 | XGBoost |
| spirit_cost_bp | 166.3186 | 167.9300 | 189.3880 | 148.8180 | ExtraTrees |
| punished_probability | 0.11992 | 0.12001 | 0.12134 | 0.08687 | ExtraTrees |
| commitment_frames | 9.5162 | 9.8477 | 9.3433 | 9.0781 | ExtraTrees |
| terminal_value | 0.003516 | 0.004774 | 0.003610 | 0.005114 | Baseline |

The CatBoost ensemble and baseline agree on 90.1% of top actions in the 222
contexts with multiple supported actions. Cross-family agreement is lower:

| Pair | Top-action agreement | Disagreements |
| --- | ---: | ---: |
| Baseline / XGBoost | 79.3% | 46 |
| Baseline / ExtraTrees | 74.3% | 57 |
| CB ensemble / XGBoost | 80.2% | 44 |
| CB ensemble / ExtraTrees | 76.6% | 52 |
| XGBoost / ExtraTrees | 79.3% | 46 |

The candidates therefore differ enough for a physical comparison; they are not
four serializations of effectively the same policy.

## Artifact paths and hashes

| HF path | Distilled policy SHA-256 |
| --- | --- |
| `lunatic30/catboost-baseline/` | `394033252789d010c60959f87b2ff7983ccd1d5ea36b3b814750ede25ea81bbc` |
| `lunatic30/catboost-ensemble5/` | `26ab9a4d4a085c55eb1b72a4030b82c2076fac57951a64a88b04c5f32e380ece` |
| `lunatic30/xgboost-hist/` | `0b5d98ea50b5279d071642184ef40284f580af4ffa75987ffca95f602a0c5436` |
| `lunatic30/extra-trees/` | `9f94103aa9a518d282438398acf07bc9aaf3c9645489832942904a1c4f542dbb` |

`lunatic30/comparison.json` contains the machine-readable held-out and pairwise
comparison.

## Recommendation and limits

ExtraTrees is the first candidate to test because it has the lowest held-out
MAE for seven of eight heads. XGBoost is second because its tail-damage estimate
is substantially better and its distilled choices differ from ExtraTrees.
The CatBoost ensemble is third, with the original CatBoost artifact retained as
the control.

This is an experiment order, not an offline promotion decision. The held-out
set contains only six complete rounds, transitions inside a round are strongly
correlated, and several outcomes are rare. A complete-round interleaved A/B
test may reverse the offline ordering.

The corpus reports `legal_actions_known: 0` for every transition. It can support
supervised outcome models for actions that were actually taken, but it cannot
support credible counterfactual CQL, IQL, or fitted-Q policy claims. More CPU
does not repair missing action coverage. The highest-value next data improvement
is controlled legal-action logging and exploration across more complete rounds,
opponents, and difficulty strata.

All four bundles passed model/distillation hash checks, exact game-build
compatibility, temporary installer validation, and the 154-test repository
suite. Promotion still requires shadow checks followed by physical rounds.
