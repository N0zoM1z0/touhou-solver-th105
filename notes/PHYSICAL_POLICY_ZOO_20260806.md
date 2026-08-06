# Lunatic policy-zoo physical screen -- 2026-08-06

## Outcome

The four distilled CPU candidates completed a fixed-Lunatic physical screen,
but **none is promoted by this experiment**. ExtraTrees and XGBoost advance to
a future matched-opponent screen. The two CatBoost variants are deprioritized.

This is not a claim that CatBoost is intrinsically unsuitable. The screen found
that the current runtime integration exposes too little of the offline policy
to distinguish the trained models reliably, while a fresh game process does
not produce the same opponent schedule for every candidate.

The final CatBoost control artifact was removed from the active runtime path.
The common online learner checkpoint was restored byte-for-byte and no game,
controller, corpus watcher, or input bridge remained running.

## Contract and provenance

- Character: Sakuya.
- Difficulty: fixed Lunatic.
- Target: 10 native terminal phases, stopping only at a complete Arena stage
  boundary. This produced 10--13 phases per candidate.
- Every candidate began from the same four online learner files. Their hashes
  were verified before and after the screen.
- Every candidate ran in a fresh game process.
- Every transition records the exact offline-policy SHA-256.
- Source after orchestration hardening: `817eb89`.
- Repository suite after the fixes: 157 tests passing, plus Linux and Windows
  byte-code compilation.

The artifact hashes were:

| Candidate | Distilled policy SHA-256 |
| --- | --- |
| ExtraTrees | `9f94103aa9a518d282438398acf07bc9aaf3c9645489832942904a1c4f542dbb` |
| XGBoost histogram | `0b5d98ea50b5279d071642184ef40284f580af4ffa75987ffca95f602a0c5436` |
| CatBoost ensemble | `26ab9a4d4a085c55eb1b72a4030b82c2076fac57951a64a88b04c5f32e380ece` |
| CatBoost baseline | `394033252789d010c60959f87b2ff7983ccd1d5ea36b3b814750ede25ea81bbc` |

## Aggregate physical results

`damage` and `self damage` are sums of option-level basis-point outcomes,
normalized by terminal phase. They measure accumulated interaction, not final
HP alone. Terminal margin is own remaining HP on a win and negative enemy
remaining HP on a loss.

| Candidate | W-L | Win rate | Punished | Damage / phase | Self damage / phase | Spirit cost / phase | Mean terminal margin | Seconds / phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees | 3-8 | 27.3% | 5.64% | 5,719 | 8,436 | 65,619 | -2,717 | 142.2 |
| XGBoost | 3-7 | 30.0% | 4.73% | 6,859 | 8,316 | 74,478 | -1,556 | 144.5 |
| CatBoost ensemble | 2-11 | 15.4% | 7.51% | 6,789 | 9,079 | 75,676 | -2,312 | 156.4 |
| CatBoost baseline | 2-8 | 20.0% | 5.39% | 7,282 | 8,872 | 83,058 | -1,876 | 154.0 |

XGBoost has the best aggregate safety and terminal-margin figures. ExtraTrees
uses the least spirit and did better on the most useful shared-opponent slice.
The CatBoost ensemble has the highest punished and self-damage rates. These are
observations, not causal model rankings, because opponent schedules differ.

## Opponent coverage and matched slices

| Candidate | Opponent results |
| --- | --- |
| ExtraTrees | `1B9C` 3-2; `22DC` 0-2; `0594` 0-2; `2074` 0-2 |
| XGBoost | `1B9C` 2-3; `1154` 1-4 |
| CatBoost ensemble | `013C` 0-2; `165C` 0-2; `1B9C` 2-5; `13D4` 0-2 |
| CatBoost baseline | `0BEC` 0-2; `1154` 1-2; `18DC` 1-2; `22DC` 0-2 |

The strongest common slice is opponent `0x006B1B9C`: ExtraTrees was 3-2,
XGBoost 2-3, and the CatBoost ensemble 2-5. Other overlaps are too small:
ExtraTrees and the baseline were both 0-2 against `22DC`; XGBoost was 1-4 and
the baseline 1-2 against `1154`.

The next comparison must therefore fix or stratify the CPU opponent instead of
relying on the fresh-process Arena schedule.

## Runtime integration finding

The artifacts each contain 946 contexts and 1,234 context-action rows. The
runtime currently consults offline outcomes only while ranking an already-legal
`space-control-*` skill set. Only 12 distilled rows can participate:

- nine `space-control-214C` rows;
- three `space-control-236B` rows;
- no supported rows for the other runtime skill candidates.

The bounded context backoff made those sparse rows reachable without crossing
difficulty, opponent, or action boundaries. Lookup coverage was:

| Candidate | Hits | Misses | Hit share | Exact hits |
| --- | ---: | ---: | ---: | ---: |
| ExtraTrees | 136 | 542 | 20.1% | 0 |
| XGBoost | 116 | 726 | 13.8% | 0 |
| CatBoost ensemble | 186 | 727 | 20.4% | 0 |
| CatBoost baseline | 179 | 604 | 22.9% | 1 |

All 12 usable rows have negative predicted utility and support-scaled
adjustments of only about `-35` to `-40`. Their distributions are:

| Candidate | Mean adjustment | Min | Max |
| --- | ---: | ---: | ---: |
| ExtraTrees | -35.37 | -35.55 | -35.27 |
| XGBoost | -35.56 | -38.57 | -35.05 |
| CatBoost ensemble | -36.70 | -39.92 | -35.67 |
| CatBoost baseline | -36.36 | -39.89 | -35.22 |

The runtime blends only 35% of this value. The typical cross-model difference
is therefore below one tactical point and is unlikely to reorder candidates.
The observed gameplay variance mostly measures opponent sampling and online
exploration, not the held-out model differences reported by the CPU trainer.

This also explains why lower held-out MAE did not translate cleanly to better
closed-loop play: most outcome heads and most context-action disagreements are
not reachable by the current runtime adapter.

## Infrastructure findings and fixes

The screen produced three reusable fixes:

1. Each transition and sanitized encounter now records the loaded offline
   artifact SHA-256 (`3c863bd`).
2. Exact-only context lookup had zero live hits in the initial pilot. A cached,
   bounded nearest-context lookup now stays within the same difficulty,
   opponent, and action, preserving the native safety gates (`ccd66c1`).
3. Windows redirected controller JSON used GBK because it contained a Chinese
   game path. The runner now forces UTF-8, accepts existing GBK output, and
   snapshots final models before parsing or export (`817eb89`).

The invalid exact-lookup pilot was kept separate from the formal screen. Its
result was not counted.

## Decision and next experiment

- Do not promote either CatBoost candidate.
- Retain ExtraTrees and XGBoost as finalists; XGBoost is the provisional safety
  leader, while ExtraTrees leads the shared `1B9C` slice and spirit economy.
- Do not claim a final winner until both use the same opponent schedule.
- Keep the active runtime free of an offline artifact until that confirmation.

The highest-value data and integration improvement remains legal candidate-set
logging. Once `legal_actions` is known, offline training can score alternatives
the controller actually had, rather than only the behavior action. A future
adapter can then apply the outcome heads to a broader native-gated candidate
set without letting an offline model invent illegal defense, movement, or skill
choices.

The next physical screen should use fixed opponents and interleave ExtraTrees
and XGBoost at complete stage boundaries. Report per-opponent confidence
intervals, but continue to let physical gameplay be the final promotion test.
