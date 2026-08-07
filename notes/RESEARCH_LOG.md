# TH105 reverse-engineering log

Evidence labels below mean:

- **Observed**: shipped instructions and/or live process state on the supported binary.
- **Inferred**: a data-flow conclusion not yet exhaustively tested across modes.
- **Hypothesized**: a candidate for a later probe.

## 2026-08-05 — launch, input, and Practice bootstrap

Supported build: `th105c.exe`, SHA-256
`49c23d9467b9927ba687ed2b873c4bc2d2f39ddadc9f55051ccf10172c0b7c11`,
fixed image base `0x00400000`.

### Engine and input roots

- **Observed** `0x006ECE7C`: current engine scene ID. At `0x0040800C`, the
  scene transition commit copies requested to current after swapping the pending
  scene controller.
- **Observed** `0x006ECE78`: requested engine scene ID.
- **Observed** `0x006ECE44`: current scene-controller pointer.
- **Observed** `0x006E62EC`: game mode; `8` is Practice.
- **Observed** `0x006ECFF0`: 256-byte DirectInput keyboard state populated by
  `IDirectInputDevice8::GetDeviceState` in the main loop.
- **Observed** `0x006E6300`: P1 input object. Bindings begin at `+0x08`, signed
  direction/button hold counters at `+0x38`, and the logical mask at `+0x62`.
- **Observed** `0x006E7520`: combined menu input object.

The profile stores DirectInput DIK indices. Extended arrows are `C8/D0/CB/CD`,
whereas Win32 `SendInput` uses scan codes `48/50/4B/4D` plus the extended flag.

### Why the injected bridge exists

- **Observed** Win32 `SendInput` made `GetAsyncKeyState` report the injected key,
  but TH105's DirectInput buffer stayed clear.
- **Observed** a temporary hook at `0x00408218`, immediately after keyboard
  `GetDeviceState`, can OR a private 256-byte DIK overlay into the native buffer.
  P1 and combined-menu counters then advance and the logical input mask changes.
- **Observed** the original hook-site bytes are `A1 DC CF 6E 00`. The controller
  suspends the process for the five-byte publication/restoration, verifies both,
  clears the overlay, restores the bytes, and frees the allocation.

### Menu route

- **Observed** scene `1`: opening story/title animation; a 35 ms Z edge skips it
  without leaking into the main-menu selection.
- **Observed** scene `2`: main menu, vtable `0x006ACCF4`; selection is controller
  field `+0x21C`. Practice is index `5`.
- **Observed** scene `3`: shared character/deck/stage selection controller.
- **Observed** scene `6`: loading transition.
- **Observed** scene `5`: battle.

A cold physical run launched PID 10668, skipped scene 1, selected main-menu index
5, confirmed default Reimu and Marisa plus the default stage, and ended in scene
5 with game mode 8. A subsequent `right@6,neutral@3,z@3,neutral@8` combo was
accepted, and the hook site was verified restored afterward.

### IDA names applied

- `engine_main_loop_poll_input_and_update_scene` (`0x004080E0`)
- `commit_pending_engine_scene` (`0x00407F80`)
- `init_direct_input` (`0x0040CDE0`)
- `create_direct_input_keyboard` (`0x0040CFA0`)
- `create_direct_input_mouse` (`0x0040D250`)
- `update_player_input_counters_from_raw` (`0x00409900`)
- `build_player_logical_input_mask` (`0x00409AD0`)
- `merge_menu_input_sources` (`0x00409C50`)
- `main_menu_update` (`0x00424AB0`)
- `main_menu_render` (`0x00424860`)
- `get_game_mode` (`0x00439860`)

### Next reverse-engineering gate

Find and type the battle-manager and both actor roots, then validate position,
velocity, facing, action/move ID, action frame, HP, spirit, card hand, hit/guard
state, and projectile ownership. The first policy should be a conservative
Easy Arcade controller that blocks high/low correctly, retreats or flies around
projectiles, and punishes only observed recovery windows.

## 2026-08-05 — corrected Arcade route and live fighter roots

- **Corrected, observed in IDA** main-menu index `1` calls
  `configure_game_mode_and_load_select(1, 0)`, but its second argument is
  replay/input routing state (`0x006E62E4`), not CPU difficulty. CPU difficulty
  is `get_game_config()+0x64` (`0x006E6B9C`) and is clamped to `0..3` by
  `CMenuConfig_ctor`: Easy, Normal, Hard, Lunatic. The controller changes it
  through Config row 0 and validates the native value after every input edge.
  Practice index `5`/mode `8` remains excluded because it has no CPU policy.
- **Observed** the character-select UI grid is not the internal character-ID
  order. Its first row is Reimu, Sakuya, Youmu, Marisa; therefore Sakuya is
  cursor slot `1`, a single Right edge from the default Reimu slot. Slot `2`
  is Youmu.
- **Observed** `0x006E6244` is the live `CBattleManagerBase`/derived pointer.
  Fighter roots are manager `+0x0C` (P1) and `+0x10` (P2).
- **Observed** fighter float position is `+0xEC/+0xF0`, velocity is
  `+0xF4/+0xF8`, facing is signed byte `+0x104`, action/sequence/pose/frame are
  16-bit fields `+0x13C/+0x13E/+0x140/+0x142`, and HP/max HP are 16-bit fields
  `+0x174/+0x176`.
- **Observed** Arcade transition `scene 5 -> scene 6` occurs between fights.
  The long-running controller now releases input during loading, reacquires the
  newly allocated manager/fighters on the next scene 5, and restarts Arcade if
  it returns to the main menu.
- **Inferred** locomotion actions occupy the low IDs; controlled samples placed
  hit/guard reactions at `60..99` and several attacks at `199+`. This is only a
  bootstrap classifier until move metadata and active/recovery frames are read.

IDA names added: `g_battle_manager`, `g_battle_scene_renderer`,
`create_arcade_battle_manager_and_renderer`, `CBattleManagerArcade_ctor`,
`CBattleManagerBase_ctor`, `arcade_battle_update_state_machine`, and
`resolve_fighter_body_collision`.

Additional IDA names/comments: `get_game_config`, `g_game_config`,
`CMenuConfig_ctor`, and the CPU-difficulty field comment at `0x006E6B9C`.

## 2026-08-06 — terminal integrity and transient projectile snapshots

- **Observed and corrected** a scene-5 projectile-list race could end the inner
  policy loop while a dead-fighter result screen was still visible. The rebuilt
  loop reported the same terminal state again, so Hard/Aya telemetry proved two
  real losses while the compact round summary contained four. A shared terminal
  tracker now requires a live `both HP > 0` observation before arming exactly
  one terminal result. Cold attachment to a dead screen is deliberately ignored.
- **Physical validation** on Hard/Remilia produced terminal sequences `1` and
  `2`; telemetry, the encounter summary, and the durable model all recorded
  exactly two losses. The previously polluted Hard/Aya round counter was
  reconciled from four to its two telemetry-proven losses without changing
  action, geometry, defense, projectile, or cancel evidence.
- **Observed and corrected** the intrusive projectile object list can mutate
  between its begin/end reads. A single mismatch formerly discarded the whole
  hot policy instance and its active outcome episode. The loop now reuses at
  most two copied snapshots, then temporarily supplies an empty set, and only
  rebuilds after 60 consecutive unavailable frames.
- **Physical validation** the same Remilia encounter ran continuously for
  18,568 policy frames across 31 list races (maximum consecutive race: four),
  then exited normally through the scene transition. No policy restart or
  reload failure occurred.
- Added a lightweight descriptive evaluator for per-opponent/difficulty
  coverage, offense/defense rates, tail damage, and confidence intervals.
  These proxy metrics are diagnostic only; complete-round physical A/B remains
  authoritative except for hard compatibility, safety, crash, and latency
  failures. Future GPU replay/artifact details are in `OFFLINE_TRAINING.md`.
- **Observed and corrected** difficulty-specific cold-start priors could contain
  legacy raw trials plus newer normalized trials. The reward switched to the
  normalized totals as soon as one v1 sample existed, while confidence still
  counted every legacy trial. Import now backfills the missing HP/spirit basis
  points and a conservative damage histogram before adding new samples. A live
  Hard/Suika checkpoint confirmed migrated rows have `normalized_samples ==
  trials` and preserve raw damage/self-damage exactly.

## 2026-08-05 — Sakuya move sources and hot policy boundary

- **Observed (binary/runtime)** Sakuya's character dispatcher at `0x004DEF70`
  reads the command mask at actor `+0x728` and selects special actions in the
  `500..599` range. It is now named
  `Sakuya_handle_input_and_select_action`; `0x004DDB20` is named
  `Sakuya_on_action_changed`.
- **Historical community reference, with input/action groups runtime-validated** the
  default move groups are `236B/C` Magic Star Sword, `214B/C` Bounce
  No-Bounce, `623B/C` Crossup Magic, and `22B/C` Vanishing Everything. A basic
  listed route is `AAA>2B>C>236B`; this is a candidate, not yet a promoted
  cancel-timing fact.
- **Observed** a 120 Hz trace captured the policy's facing-relative `236B`
  sequence and command mask `0x202` selecting action `500`; labeled policy
  trials likewise pair `236C` with `501` and `214B/C` with `520/521`. The
  default mapping is therefore `236B/C=500/501`, `214B/C=520/521`, and the
  remaining `623B/C=540/541` group. Frame-data flag `0x20` is present in
  idle/cancellable recovery frames and absent through the locked portion of
  action 500, matching the dispatcher's input gate. Action appearance alone is
  not treated as command identity.
- **Implemented** the long-lived scene/input shell hot-loads a versioned policy
  source by SHA-256 every 30 combat frames. Invalid source or a first-decision
  exception rolls back to the last good generation. JSONL heartbeats preserve
  action/pose/frame, HP, projectiles, intent, reload status and histograms.
- **Implemented** `th105_hazard.dll` ABI v1 evaluates many constant-velocity
  player candidates against all currently active projectiles over a bounded
  horizon. `hazard.py` retains a Python reference/fallback; Windows smoke
  testing selected the native backend and agreed on a controlled collision/
  retreat example.
- **Observed (dispatcher data flow)** actor `+0x158` is the current frame-data
  pointer and its `+0x4C` flags gate input/cancel processing (`0x20` and
  `0x200000` branches). Actor `+0x180` participates in hit/collision state,
  `+0x482` is the spirit threshold used before shots/skills, and `+0x728` is the
  decoded command mask. These are now included in fighter telemetry for
  action-synchronous combo tuning.

## 2026-08-05 — native frame collision geometry and guard re-arming

- **Observed (IDA/runtime)** `transform_local_aabb_to_world` (`0x0046ACD0`)
  mirrors local X when facing is not `1`, adds actor X at `+0xEC`, and converts
  local Y using actor Y at `+0xF0`. `resolve_fighter_body_collision`
  (`0x0046C290`) consumes the current frame pointer at actor `+0x158` and the
  single body/push rectangle at frame `+0x54`.
- **Observed (runtime)** frame `+0x5C/+0x60/+0x64` and
  `+0x6C/+0x70/+0x74` are bounded MSVC-vector triples of four-int local
  rectangles. Idle Sakuya had one first-vector rectangle and no second-vector
  rectangles. Active projectile frames had second-vector attack rectangles;
  destructible projectiles could have the same rectangle in both vectors.
- **Corroborated (public reverse-engineered layout)** SokuDev/SokuLib commit
  `0794becf1f578b32329604483be2112fe0afd4ee` names these fields
  `hurtBoxes` and `attackBoxes`, with `collisionBox` at `+0x54` and
  `AttackFlags` at `+0x50`. Its flag layout identifies bit `0x2` as a mid hit
  and `0x4` as a low hit. This is supporting evidence; the exact TH105 binary
  and live values remain the authority for this project.
- **Implemented** the sensing shell reads hurt/attack vectors every combat
  frame. The policy transforms every active projectile attack rectangle to a
  world-space hazard; an empty attack vector is an inactive animation frame,
  not a collision threat. Player size/center now comes from the current native
  hurt-box envelope rather than the former fixed `18x42` estimate.
- **Implemented** native hazard ABI v5 adds candidate center offsets, allowing
  crouch to model both its reduced height and downward center after startup.
  A 20-frame guard-contact latch plus a 12-frame block/hit-stun latch keeps
  defense armed across gaps in multi-hit pressure. Unambiguous native mid/low
  flags override the learned guard guess; the learner remains the fallback for
  ambiguous frames.

## 2026-08-05 — read-only human offense demonstrations

- **Implemented** `learn-human` refuses an installed input hook and performs no
  focus or input calls. It observes native Sakuya action/sequence transitions,
  command masks, matchup/context, hit startup, damage, self-damage, spirit use,
  duration, direct cancel edges, and complete offensive chains.
- **Implemented** physical inputs are normalized to `toward/back`, reduced to
  run-length patterns, and capped per action/chain. Outcome tables contain only
  bounded sufficient statistics: at most 32 patterns per row, 128 chain rows
  per context, and 1024 cancel edges per matchup. No 60 FPS raw frame corpus is
  retained.
- **Implemented** successful human chains can seed autoplay only in a confirmed
  opponent reaction window. Replay validates allowed keys/run lengths, mirrors
  directions using current facing, preserves 200 spirit, and remains subject to
  the ordinary hit-confirm abort, threat cancellation, and online outcome
  learner. A merely inferred recovery window still permits only one close-A
  probe, not a full demonstrated or built-in combo.
- **Observed (first physical Sakuya session)** 465 seconds of read-only play
  covered three opponent vtables, 329 action trials, 180 offensive-chain trials,
  and 94 distinct action transitions. Only aggregates and bounded normalized
  input patterns were retained; no raw frame stream entered the training data.
- **Implemented** the offline human compiler ranks chains by connected damage,
  self-damage, spirit cost, duration, and repeated-pattern support. It rejects
  connection rates below 50%, spends that violate the 200-spirit reserve, and
  trades whose average self-damage exceeds both 100 and 35% of dealt damage.
  The resulting read-mostly artifact had 47 candidates across 33 contexts, with
  no more than eight candidates considered by an online decision.
- **Implemented** compact live policy metrics report loaded human context and
  candidate counts, total replay attempts, and the last selected signature.
  This makes demonstrated-policy use auditable without retaining raw combat
  frames.
- **Observed/implemented** rapid physical direction changes can expose one-frame
  simultaneous opposites in the raw DirectInput buffer. Collection now records
  those horizontal or vertical pairs as neutral. The offline compiler applies
  the same canonicalization to earlier sessions and merges their pattern
  support, so replay never emits `toward+back` or `up+down` chords.
- **Observed (IDA RTTI)** all 15 concrete fighter vtables are uniquely named in
  the supported binary. Runtime telemetry now resolves them to roster names;
  the first demonstration matchups were Reimu (`0x006B013C`), Yuyuko
  (`0x006B165C`), Komachi (`0x006B2074`), and Aya (`0x006B22DC`). Reisen is
  named `Udonge` in RTTI and has fighter vtable `0x006B1E3C`.
- **Observed/inferred** main-menu index 2 calls the now-renamed
  `configure_game_mode_and_load_select` with `(2, 1)` and enters the shared
  selection scene. It is the candidate VS CPU/Arena curriculum route, but P2
  cursor offsets and CPU ownership still require a live calibration before the
  controller may use it.
- **Observed (IDA)** `CSelect_update_character_phase` iterates two selection UI
  records with a `0x28` stride and writes two player-result records with a
  `0x20` stride. This places the validated P2 cursor/selection mirrors at
  controller `+0x134/+0x138`, corresponding to P1 `+0x10C/+0x110`. Read-only
  access is implemented; navigation remains gated on a live game-mode-2 probe.
- **Implemented** human pattern statistics now bind each exact normalized input
  trace to its own trials, connections, damage, self-damage, spirit cost, and
  duration. This separates Sakuya's facing-relative advancing attacks such as
  `toward+A+X`, `toward+A+C`, and `toward+A+Z` from retreating variants even
  when they resolve to the same action/sequence signature. Compiled candidates
  carry a stable short pattern ID, and online outcome learning keeps variants
  distinct. Legacy count-only patterns remain readable and are progressively
  upgraded by later demonstrations.
- **Implemented** confirmed-punish replay now considers human demonstrations out
  to the full mid-range bucket (`<260`) only when the trace contains a
  facing-relative advance followed within six frames by Z/X/C. Ordinary human
  strings and built-in grounded combos retain their close-range gates. This
  exposes Sakuya's advancing attacks without turning mid-range recovery guesses
  into blind combo commitments.
- **Observed/implemented** full demonstrated chains could not create their own
  first hit because they were correctly gated behind confirmed reaction. A
  neutral-state replay now extracts only the bounded approach/attack edge. In
  live Easy Arcade this selected Sakuya action `418` and reduced Marisa from
  10000 to 8636 without committing the remainder of the human chain.
- **Implemented** neutral exploration no longer depends on human data. Five
  short facing-relative probes (`toward+Z`, `toward+A+Z/X/C`, and rising-Z) join
  demonstrated candidates in a per-context UCB-style bandit. Reward combines
  enemy damage, 1.5x self-damage penalty, spirit cost, and commitment; every
  bounded candidate is sampled optimistically, while repeatedly losing trades
  are retired. Online state remains fixed-size sufficient statistics per
  opponent/context/candidate.
- **Observed/implemented** native mid/low attack flags are strong guard priors,
  not infallible labels. When a response has taken real damage for an exact
  action/sequence signature, the learned alternative may override the native
  prior. This addresses repeated Youmu pressure where a nominal high guard lost
  724 and 531 HP on separate episodes.
- **Implemented** `auto-arcade --continuous` and
  `run_th105_overnight.bat` provide an unattended curriculum. Encounter
  boundaries checkpoint and compile cumulative per-opponent timing, projectile,
  defense, and offense aggregates; bounded telemetry continues rotating.
- **Implemented/validated** native hazard ABI 6 adds acceleration to every
  candidate movement path. Jump and super-jump candidates now follow a gravity
  parabola through ascent, apex, and descent instead of being projected upward
  forever; dash and controlled flight remain linear. A Windows-side parity
  probe loaded the rebuilt native DLL and matched the Python reference to
  `2.8e-7` clearance units on mixed ballistic/linear candidates.
- **Implemented** long encounters now atomically checkpoint all bounded action,
  projectile, defense, and offense aggregates every 10 seconds as well as at
  the encounter boundary. An interrupted overnight run loses at most one
  checkpoint interval; the read-mostly policy compiler still runs only between
  encounters to keep the per-frame loop free of compilation work.
- **Implemented** a character-agnostic attack geometry learner keyed by native
  `action_id:sequence`, with bounded per-pose rows. It records local attack-box
  envelopes plus first/last active elapsed frames, persists them per opponent,
  and compiles compact read-mostly lookups. No move names, matchup rules, or
  manually entered hitboxes participate.
- **Implemented** native hazard ABI 7 adds delayed active intervals to hazards.
  On repeat encounters, an opponent melee action can therefore contribute its
  previously observed future hitbox during startup. All legal guard/movement
  responses remain available to the learner, but geometrically colliding
  movement is pruned before bandit selection and is never credited as though a
  different response had been executed.
- **Observed (live Easy Arcade, Youmu)** after a cold encounter began, the
  learner accumulated 23 poses and 15 active-box observations for two early
  action/sequence rows. During a later action `305:0` frame where the current
  native attack vector was still empty, the learned model supplied one delayed
  melee hazard to the native solver. All evaluated paths were reported safe at
  that particular spacing, so no movement was falsely forced.
- **Observed/fixed** scene transitions can briefly start a second battle shell
  with already-dead fighters. Because it executes zero policy decisions, its
  models were never seeded; encounter-end persistence could then overwrite the
  just-learned character state with empty dictionaries. Persistence now
  requires at least one valid decision frame, making transition-only shells
  read-only and preserving cumulative overnight learning.
- **Observed/fixed** a second transition path invalidated the fighter or object
  manager while the scene byte still reported battle, raising to the outer
  recovery loop before encounter-end persistence. Bounded fighter/projectile
  read failures now terminate the encounter normally, checkpoint the valid
  decision frames, and then let the existing outer loop reacquire fresh actors.
- **Observed/fixed** hot reload transferred defense/offense/projectile state but
  not cumulative temporal action profiles, then allowed encounter-start disk
  priors to overwrite the transferred state on the new generation's first
  decision. Export/import now carries both opponent and own temporal profiles
  and marks all transferred model families as already seeded; disk priors are
  used only on a genuinely fresh policy.
- **Implemented** a bounded native cancel graph records only accepted Sakuya
  offensive transitions. Attribution starts on a Z/X/C rising edge, retains its
  facing-normalized chord and exact source action/sequence/pose/frame for at
  most six frames (covering the command buffer), and binds it to the first
  resulting offensive action transition. Automatic animation transitions and
  idle/reaction exits are rejected. During a real enemy hit reaction, only an
  edge observed at the current pose/timing may be explored; its continuation
  reward is learned online rather than assigned from a guide.
- **Implemented** reward v1 separates durable native outcomes from their
  scalarization. Attack and defense rows now retain raw plus max-normalized
  damage/resource components and compact eight-bin downside histograms. The
  scorer penalizes whiffs, losing trades, commitment, resource use, and
  high-damage tails.
- **Implemented** the battle shell reports every native round result exactly
  once. Recent completed offense options receive a small decayed eligibility
  credit, while immediate exchange damage remains dominant. Round aggregates
  and reward version persist, and legacy raw-only rows remain readable.
- **Specified** independent corpus schema, reward, and compiled-artifact
  versions in `notes/CORPUS_SCHEMA.md`. Raw sufficient statistics are durable;
  reward/Q/ranks are disposable derivatives, keeping the corpus reusable by a
  later offline RL or server-trained model.
- **Observed/fixed (Patchouli Lunatic, 2026-08-07)** coverage counters were
  exported by the plugin but omitted from durable per-opponent knowledge. They
  are now checkpointed, restored, and keyed by a bounded family/distance/
  altitude/phase scope. Exact learned-cancel edges sharing one input chord use
  one coverage counter, while their outcome rows and raw legal set remain
  distinct. Legacy full-context counters are discarded on import.
- **Implemented** sparse exact action values now shrink through per-opponent
  family and motion/cancel-input sufficient statistics. UCB uses that effective
  support with a finite pseudo-count and a capped bonus instead of assigning
  every unseen exact variant utility 2000. Explicit legal-set exploration
  decays from the collection rate toward a small family-specific floor.
- **Implemented** safe neutral play is an n-step option bandit over `defend`,
  `reposition`, `approach`, `attack`, and `punish`. Native hit reaction,
  projectile/melee hazard, guard-chain, spell, and commitment branches retain
  veto priority. Both the high-level legal option set/propensity and exact
  micro-action set/propensity are stored in each transition.
- **Implemented/validated live** completed offense and option rows retain a
  360-frame projectile-weighted eligibility trace for delayed damage. The first
  live v5 Patchouli/Komachi round completed with the opponent at 53.7% HP;
  60-second checkpoints contained both coverage and option models, the next
  Arcade opponent loaded without a policy error, and all five high-level
  options appeared in the fresh transition stream.
- **Implemented (v6, 2026-08-07)** high-level option and concrete-action
  outcomes are conditioned on the opponent's exact offensive `action_id`,
  learned strike/projectile/hybrid/spell family, action phase, distance,
  relative height, and corner state. Sparse observations back off through at
  most six context levels and the existing action-side hierarchy; native hazard
  vetoes remain authoritative.
- **Implemented** reusable autonomous-grammar-v4 transition shards can be
  replayed into fresh v6 online weights. New rows store the exact runtime
  `learning_context`; old rows use conservative native-state/profile inference.
  Replay keeps separate damage, self-damage, spirit, commitment, effect, and
  terminal components and emits a separate artifact before installation.
