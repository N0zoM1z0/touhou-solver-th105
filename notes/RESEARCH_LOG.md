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

- **Observed** main-menu index `1` calls `sub_43A560(1, 0)`: game mode `1`
  (Arcade) with difficulty `0` (Easy). Practice index `5`/mode `8` is no
  longer the autoplay route because it has no CPU policy.
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
