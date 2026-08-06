# Lunatic30 CatBoost baseline -- 2026-08-06

## Immutable inputs and outputs

- Training code: Git commit `4479e41a64a8eeee60874b7471783bdb73f20ab5`.
- Private dataset: `Joh1rreq/touhou-solver-th105-corpus`, revision
  `fbd13cb9c900c81d3623a9c7ac515896e2f00800`.
- Dataset source code: commit
  `8bbcf7e6f456f6ecad26bab003da13b9f468c234`; the CPU feature and trainer code
  did not change between that commit and the training commit.
- Private model repository: `Joh1rreq/touhou-solver-th105-policy-zoo`, revision
  `dd06f180049998d094544451fdb1db4d4efa832e`.
- Bundle path: `lunatic30/catboost-baseline/`.

The model revision was downloaded again after upload and compared byte-for-byte
with the local bundle. The repository was also confirmed private at that exact
revision.

## Corpus validation

- 15,039 unique transitions in 30 complete episodes.
- 12,318 training rows and 2,721 held-out rows, split by complete chronological
  episodes rather than adjacent random rows.
- 9 Lunatic opponents; 6 wins and 24 losses.
- No duplicate transition IDs, step gaps, or incomplete episode terminals.
- All 10 exported dataset artifact sizes and SHA-256 hashes matched the corpus
  manifest.
- Transition SHA-256:
  `4df690f49f10a74aeeb09ee4cd67e396dd53d72701b0e0a66f8e87c5e8dbad84`.
- Corpus manifest SHA-256:
  `eb555d702a614f396fafce4134e67e4039240b92e7dadcd0c32ddc10528de071`.

## Training run

Environment: CPython 3.14.6, CatBoost 1.2.10, huggingface_hub 1.26.0,
96 physical AMD EPYC 9654 cores.

```bash
python scripts/train_th105_cpu.py \
  --input dataset-lunatic30/data/transitions.jsonl.gz \
  --corpus-manifest dataset-lunatic30/manifest.json \
  --output policy-lunatic30-cpu \
  --threads 96 --iterations 600 --depth 8
```

Wall time was 29.995 seconds. The eight outcome heads used early stopping on
the held-out episodes.

| Outcome head | Best iteration | Held-out MAE | Held-out RMSE |
| --- | ---: | ---: | ---: |
| damage_bp | 298 | 17.4334 | 61.1549 |
| connection_probability | 313 | 0.04410 | 0.14137 |
| self_damage_bp | 135 | 40.2131 | 89.5913 |
| self_damage_p90_bp | 66 | 42.8672 | 108.6491 |
| spirit_cost_bp | 155 | 166.3186 | 383.5038 |
| punished_probability | 210 | 0.11992 | 0.23605 |
| commitment_frames | 584 | 9.5162 | 15.3011 |
| terminal_value | 3 | 0.003516 | 0.046744 |

The resulting runtime table contains 946 contexts and 1,234 context-actions.
Its SHA-256 is
`394033252789d010c60959f87b2ff7983ccd1d5ea36b3b814750ede25ea81bbc`.
All model hashes, the distilled hash, exact game build compatibility, and the
installer path were verified. The repository test suite passed 152/152 tests.

## Retrieve and validate

```bash
hf download Joh1rreq/touhou-solver-th105-policy-zoo \
  --revision dd06f180049998d094544451fdb1db4d4efa832e \
  --include 'lunatic30/catboost-baseline/**' \
  --local-dir policy-zoo

python scripts/install_th105_offline_policy.py \
  --bundle policy-zoo/lunatic30/catboost-baseline
```

Installation was tested only against a temporary runtime directory on the
training server. This record does not claim that the candidate improves live
play. Shadow compatibility checks and complete physical round A/B evaluation
remain required before promotion.
