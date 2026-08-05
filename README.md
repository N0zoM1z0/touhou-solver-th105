# touhou-solver-th105

An experimental, explainable autoplay engine for Touhou 10.5 / Scarlet Weather
Rhapsody. It launches and navigates an Easy Arcade campaign with Sakuya, senses
native combat state, drives the game through a reversible background DirectInput
bridge, and learns compact per-character offense and defense models from play.

This repository does not contain the game, game assets, or a patched executable.
It supports one exact `th105c.exe` build identified by SHA-256 in
`scripts/th105/constants.py`.

## Current capabilities

- Native scene validation and Easy Arcade/Sakuya menu navigation.
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

## Running

The controller runs under Windows Python because it uses Win32 process and input
APIs directly. From a Windows prompt:

```bat
run_th105_auto_arcade.bat
```

The launcher is equivalent to
`auto-arcade --launch --p1-character sakuya --battle-seconds 300`. The injected
bridge drives TH105's private DirectInput buffer, so autoplay continues in the
background while another window has focus. Pass `--foreground-only` to restore
strict foreground ownership. Round-end and Arcade dialogue screens receive
sparse Z edges so the campaign can continue.

For an unattended learning run, use `run_th105_overnight.bat`, equivalent to
`auto-arcade --launch --p1-character sakuya --continuous`. Each completed
encounter atomically checkpoints its per-opponent model, with an additional
10-second crash-recovery checkpoint; Ctrl+C restores the input bridge before
exit. Neutral offense uses a bounded contextual bandit: it
tries short `toward+Z`, `toward+A+Z/X/C`, and rising-Z probes, measures damage,
self-damage, spirit cost, and commitment, exploits good results, and retains a
shrinking exploration bonus. This works without a human corpus; demonstrations
only improve cold-start ordering.

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
```

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
