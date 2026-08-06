# touhou-solver-th105

An experimental, explainable autoplay engine for Touhou 10.5 / Scarlet Weather
Rhapsody. It launches and navigates Arcade campaigns with Sakuya, senses
native combat state, drives the game through a reversible background DirectInput
bridge, and learns compact per-character offense and defense models from play.

This repository does not contain the game, game assets, or a patched executable.
It supports one exact `th105c.exe` build identified by SHA-256 in
`scripts/th105/constants.py`.

## Current capabilities

- Native scene validation, Sakuya menu navigation, and fixed or automatically
  promoted `Easy/Normal/Hard/Lunatic` CPU difficulty.
- Background-capable injected input with exact-byte validation, orphan recovery,
  key release on exit, and post-cleanup verification.
- Hot-reloadable combat policy; a bad generation rolls back without stopping the
  campaign.
- Fighter/action/pose/frame, HP, spirit, facing, velocity, native per-frame
  hurt/attack boxes, and active projectile sensing from read-only process memory.
- Bounded per-opponent `action/sequence/pose` attack-box learning; future melee
  active windows are projected from native observations rather than character
  move-name rules.
- A C++ finite-horizon projectile evaluator with exact current-frame AABBs,
  projectile and movement acceleration, movement startup, graze windows, and a
  Python parity fallback.
- Per-opponent action timing, projectile envelope, guard response, and offense
  outcome learning using bounded sufficient statistics and a lightweight
  contextual-bandit explorer rather than a neural network or heavyweight RL
  runtime.
- Versioned, dimensionless reward over native outcome components, including
  exchange loss, resource/whiff cost, downside-tail risk, and lightly weighted
  round credit. Reward changes require recompilation, not recollection.
- Between-encounter model compilation and bounded gzip telemetry rotation.

The project is research-stage. It is not yet expected to clear Arcade reliably.

## Learning architecture

The online loop stays deliberately small:

1. identify the opponent by native fighter vtable and the move by action/sequence;
2. update constant-size counters and running means from hits, blocks, resource
   changes, projectile births, and action transitions;
3. enumerate a small legal response set and run a short native trajectory forecast;
4. choose the lowest-risk defense or highest-utility safe punish.

Offense and defense keep separate outcome models. A native-state tactical gate
decides whether the present frame is a confirmed threat, punish window, or
neutral state; the corresponding learner chooses the response and timing. CPU
difficulty is part of the opponent model key, so faster Hard/Lunatic behavior
does not get averaged into Easy timing statistics.

The framework supplies legal inputs and safety constraints, but it does not
encode matchup tactics. Native geometry removes physically impossible movement
responses; a contextual outcome learner decides which remaining defense,
attack, and counter works for each opponent and state. Human demonstrations and
guide-derived routes are removable cold-start priors, not immutable decisions.
Accepted offensive transitions are also learned directly from play: an attack
button rising edge is retained for at most six frames, then bound to the native
`action/sequence/pose/frame -> action/sequence` transition it actually caused.
Only observed cancel edges can be replayed, and their continuations are ranked
by the same damage/self-damage/resource/commitment outcomes as other actions.

At encounter boundaries, cumulative models are atomically written to
`runtime/th105_opponent_models.json` and compiled into the read-mostly
`runtime/th105_compiled_policy.json`. Raw telemetry is not a training database:
it rotates at a fixed size and keeps only a few compressed diagnostic windows.
The durable data tiers and forward-compatibility contract are specified in
[`notes/CORPUS_SCHEMA.md`](notes/CORPUS_SCHEMA.md).

## Private offline artifacts

Training data and derived policies are intentionally stored outside this source
repository. Access requires a Hugging Face token with permission to both private
repositories.

| Content | Repository | Verified revision |
| --- | --- | --- |
| Lunatic30 transition corpus | [`Joh1rreq/touhou-solver-th105-corpus`](https://huggingface.co/datasets/Joh1rreq/touhou-solver-th105-corpus) | `fbd13cb9c900c81d3623a9c7ac515896e2f00800` |
| Four trained policy bundles | [`Joh1rreq/touhou-solver-th105-policy-zoo`](https://huggingface.co/Joh1rreq/touhou-solver-th105-policy-zoo) | `ccc35480bfe2c73437b5a9c114c4911dcd80101d` |

After cloning or pulling this repository on a training machine, install the Hub
client, authenticate, and download immutable snapshots:

```bash
python -m pip install -r requirements-cpu-train-zoo.txt
hf auth login

hf download Joh1rreq/touhou-solver-th105-corpus \
  --repo-type dataset \
  --revision fbd13cb9c900c81d3623a9c7ac515896e2f00800 \
  --local-dir dataset-lunatic30

hf download Joh1rreq/touhou-solver-th105-policy-zoo \
  --revision ccc35480bfe2c73437b5a9c114c4911dcd80101d \
  --include 'lunatic30/**' \
  --local-dir policy-zoo
```

The dataset then lives at
`dataset-lunatic30/data/transitions.jsonl.gz`. The model snapshot contains the
following runtime-compatible candidates:

```text
policy-zoo/lunatic30/catboost-baseline/
policy-zoo/lunatic30/catboost-ensemble5/
policy-zoo/lunatic30/xgboost-hist/
policy-zoo/lunatic30/extra-trees/
policy-zoo/lunatic30/comparison.json
```

For example, validate and install the ExtraTrees candidate with:

```bash
python scripts/install_th105_offline_policy.py \
  --bundle policy-zoo/lunatic30/extra-trees
```

This writes the distilled runtime policy and its manifest under `runtime/`; it
does not install the corpus or the training-library models into the live loop. The
complete provenance, held-out comparison, artifact hashes, and recommended
physical test order are recorded in
[`notes/TRAINING_LUNATIC30_POLICY_ZOO_20260806.md`](notes/TRAINING_LUNATIC30_POLICY_ZOO_20260806.md).

## Running

The controller runs under Windows Python because it uses Win32 process and input
APIs directly. From a Windows prompt:

```bat
run_th105_auto_arcade.bat
```

The launcher is equivalent to
`auto-arcade --launch --p1-character sakuya --battle-seconds 300`. The default
`--difficulty curriculum` promotes one level after sustained wins and demotes
one level after sustained losses; pass `--difficulty easy`, `normal`, `hard`, or
`lunatic` for a fixed campaign, or `--difficulty cycle` to rotate
Easy→Normal→Hard→Lunatic after a fixed quota of native terminal rounds. The
weighted quotas default to 2/4/8/10 respectively, assigning 75% of training
rounds to Hard and Lunatic. Override them with
`--cycle-round-quotas EASY,NORMAL,HARD,LUNATIC`, or use
`--rounds-per-difficulty N` for an equal quota.
Pass `--round-limit N` to stop at the first complete Arena match boundary at or
after N native terminal rounds; the reported actual count can exceed N by at
most the remainder of the current match.
Learned observations are stored under separate opponent-and-difficulty keys. The injected
bridge drives TH105's private DirectInput buffer, so autoplay continues in the
background while another window has focus. Pass `--foreground-only` to restore
strict foreground ownership. Round-end and Arcade dialogue screens receive
sparse Z edges so the campaign can continue.

For an unattended learning run, use `run_th105_overnight.bat`, equivalent to
`auto-arcade --launch --p1-character sakuya --difficulty cycle
--cycle-round-quotas 2,4,8,10 --continuous`. Each completed
encounter atomically checkpoints its per-opponent model, with an additional
60-second crash-recovery checkpoint; Ctrl+C restores the input bridge before
exit. A process lock rejects a second input controller before it can compete
for keys or model checkpoints. Neutral offense uses a bounded contextual bandit: it
tries short `toward+Z`, `toward+A+Z/X/C`, and rising-Z probes, measures damage,
self-damage, spirit cost, and commitment, exploits good results, and retains a
shrinking exploration bonus. This works without a human corpus; demonstrations
only improve cold-start ordering.

Event-aligned intent transitions are staged in the bounded, compressed
`runtime/th105_transitions.jsonl` corpus with raw outcome components and
versioned state/action schemas; this does not alter the live decision path.

Training permits bounded skill probes in safe far-neutral states and during a
learned enemy recovery window long enough to cover the move's startup. The same
native projectile forecast and resource checks remain active, while observed
punishes down-rank bad probes automatically.

The resident controller hot-reloads `scripts/th105/policies/adaptive.py` every
30 combat frames. A bad edit keeps the last known-good generation. Compact live
state is written to `runtime/th105_live.jsonl`; override both paths with
`--policy-plugin` and `--telemetry-path`.

Build the optional native hazard kernel from WSL:

```bash
./build_th105_native.sh
```

Windows autoplay loads `build/th105_hazard.dll` automatically and falls back to
the Python parity implementation if it is absent.

Run the unit suite and optional between-encounter compiler with:

```bash
python -m unittest discover -s tests -v
python scripts/train_th105_models.py
python scripts/evaluate_th105_learning.py
```

The evaluator reports per-opponent/difficulty coverage, offense/defense rates,
tail damage, and Wilson confidence intervals from durable evidence. It is a
descriptive monitoring gate, not a substitute for complete-round A/B testing.
The multi-core CPU corpus, artifact, and promotion protocol is specified in
[`notes/OFFLINE_TRAINING.md`](notes/OFFLINE_TRAINING.md).

Export one completed controller session as a deterministic, privacy-scanned
dataset snapshot with:

```bash
python scripts/export_th105_session.py \
  --session-id SESSION_ID --started-at UNIX_START --ended-at UNIX_END \
  --experiment-name lunatic30 --target-rounds 30 \
  --baseline-dir runtime/experiments/lunatic30/baseline \
  --output runtime/experiments/lunatic30/dataset
```

The export contains gzip JSONL transitions, sanitized terminal summaries,
baseline/final compact models, a dataset card, independent schema/reward
versions, and SHA-256 hashes. It rejects credentials and local filesystem paths.
For indefinite collection, run `scripts/archive_th105_corpus.py`; it copies
completed gzip rotations into `runtime/corpus_archive` under decompressed-content
hashes, so bounded live telemetry cannot discard offline training evidence.
On a CPU training server, install `requirements-cpu-train.txt` and train the
first multi-head outcome model with:

```bash
python scripts/train_th105_cpu.py \
  --input dataset-lunatic30/data/transitions.jsonl.gz \
  --corpus-manifest dataset-lunatic30/manifest.json --output policy-lunatic30 \
  --threads 96
```

The trainer uses CPU-only categorical boosted trees for separate damage,
self-damage/tail, spirit, punish, and terminal heads, then distills predictions
into a compact context/action table. Gameplay reward weights are applied only
when the runtime ranks safe actions: changing the offense/defense/risk trade-off
re-scores the same outcome artifact instead of requiring corpus recollection or
outcome-model retraining. Training algorithm and runtime artifact are decoupled:
native legality/hazard gates remain mandatory, while unknown or unsupported
contexts fall back to the online contextual bandit. The versioning and A/B
contract is documented in
[`notes/OFFLINE_TRAINING.md`](notes/OFFLINE_TRAINING.md#outcome-learning-is-separate-from-gameplay-preference).

To compare several CPU model families without changing the runtime contract,
install the policy-zoo dependencies and train each candidate against the same
chronological complete-episode split:

```bash
pip install -r requirements-cpu-train-zoo.txt

python scripts/train_th105_zoo.py --algorithm catboost-ensemble \
  --input dataset-lunatic30/data/transitions.jsonl.gz \
  --corpus-manifest dataset-lunatic30/manifest.json --output policy-catboost-ensemble \
  --threads 32 --iterations 900 --depth 8 --members 5

python scripts/train_th105_zoo.py --algorithm xgboost \
  --input dataset-lunatic30/data/transitions.jsonl.gz \
  --corpus-manifest dataset-lunatic30/manifest.json --output policy-xgboost \
  --threads 32 --iterations 1200 --depth 8

python scripts/train_th105_zoo.py --algorithm extra-trees \
  --input dataset-lunatic30/data/transitions.jsonl.gz \
  --corpus-manifest dataset-lunatic30/manifest.json --output policy-extra-trees \
  --threads 32 --trees 512 --depth 18

python scripts/compare_th105_policy_bundles.py \
  policy-catboost-ensemble policy-xgboost policy-extra-trees
```

The comparison reports held-out metrics per outcome head and pairwise distilled
top-action agreement. A lower transition-level error is not a promotion gate:
the final choice still requires shadow validation and complete physical rounds.
The current corpus has no known legal-action set, so it cannot support credible
counterfactual CQL/IQL-style offline-policy claims; the alternative trainers
remain outcome models behind the native legal and hazard gates.

Install a returned bundle only after its manifest, model hash, and exact game
build pass validation:

```bash
python scripts/install_th105_offline_policy.py --bundle policy-lunatic30
```

The installer atomically writes `runtime/th105_offline_policy.json`. A battle
loads it at the next encounter boundary. Initially it only blends supported
offline outcomes into already-safe skill ranking; it cannot create a legal
action or bypass native hazard, range, resource, startup, or recovery gates.

Every option transition and sanitized encounter summary records the loaded
offline-policy SHA-256. Physical A/B results therefore remain attributable even
when several distilled artifacts are installed into the same runtime path.

Run a fair four-candidate physical screen with a fixed Lunatic difficulty and
the same online-learner checkpoint for every candidate:

```bash
python scripts/run_th105_policy_ab.py \
  --rounds 10 \
  --output runtime/experiments/policy-zoo-physical-20260806
```

Each candidate runs in a fresh game process. Its distilled bundle is installed
atomically, the common online baseline is restored before play, and controller
output, session transitions, terminal summaries, plus final learned models are
kept below its experiment directory. The common online baseline is restored
again after the complete screen.

### Learning from human Sakuya play

Stop `auto-arcade`, leave the exact supported game running, then start the
read-only demonstration recorder from Windows Python:

```bat
run_th105_learn_human.bat
```

Play normally and press Ctrl+C when finished. This mode refuses to start while
the injected input bridge is present; it never focuses the window or writes to
the game. It stores bounded sufficient statistics plus facing-normalized,
run-length-encoded input patterns in
`runtime/th105_human_demonstrations.json`. The autoplay policy loads successful
matchup/context chains as priors for confirmed punish windows, then continues
scoring their real autoplay outcomes. `scripts/train_th105_models.py` compiles
the aggregate into `runtime/th105_human_policy.json`: each context keeps at most
eight ranked candidates, reserves 200 spirit, and rejects low-connect or
high-self-damage trades before the online loop sees them. Failed or unsafe
demonstrations are not blindly preferred. Exact facing-relative directional
patterns are scored separately, so Sakuya's advancing `toward+A+Z/X/C` routes
do not inherit outcomes from retreating or non-directional variants.

Useful Windows diagnostics:

```bat
python scripts\run_th105_agent.py probe
python scripts\run_th105_agent.py tap down z
python scripts\run_th105_agent.py hold right+z --seconds 0.2
python scripts\run_th105_agent.py combo "right@6,right+z@3,neutral@8"
python scripts\run_th105_agent.py fight --seconds 60
python scripts\run_th105_agent.py release
```

The supported binary identity and default game directory are recorded in
`scripts/th105/constants.py`. Override the directory with `TH105_GAME_DIR`.

## Safety and scope

The controller refuses unknown executable hashes. Runtime modification is limited
to the explicitly selected input bridge; sensing is read-only. The ordinary
launcher never distributes or rewrites game files. Reverse-engineering evidence
and confidence levels are recorded in `notes/RESEARCH_LOG.md` so offsets and
inferred semantics remain auditable.
