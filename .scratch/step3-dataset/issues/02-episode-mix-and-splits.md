# Episode mix, dataset size and splits

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What exactly gets recorded, how much of it, and how is it split?

- Size: how many episodes ("a few thousand"), episode length (900 ticks, or shorter episodes for more start diversity per tick)?
- Mix: balance across Instruction × Direction (6 combinations) — equal counts, or weighted (e.g. more `any` for the multimodality experiment)? Does Direction stay random from the seed or is it assigned?
- Start randomisation: are Step 1's `reset` ranges (any Station, ±1 m, ±10°, 0–6 m/s) enough, or does Step 3 widen any of them? What does "randomise speed" mean beyond the start speed, given the expert's speed profile is fixed?
- Expert style knobs: confirm `a_lat_plan`/`v_cap` fixed (Step 2 hidden-variable caveat); noise at Step 2 defaults?
- Splits: train/val/test by episode (never by window), proportions, and **disjoint seed ranges** from each other and from Step 9 closed-loop evaluation seeds and Step 2 `eval` seeds 0–999.
- Crashes: the expert doesn't crash, but should the recorder keep or reject a crashed episode if one ever happens?

Check: a table of the recorded set (Instruction × Direction × count, seed ranges per split, episode length, expert params) and the total size in episodes, ticks and windows.

## Comments
- Q1 size and length → **6000 episodes × 300 ticks** (20 s, about 1.4 laps, about 1.15 Passages each), before splitting: 1.8M ticks, about 1M usable (clean-future) windows. Same tick budget as 2000 × 900, but 3× more random starts; laps 2–4 of a long episode mostly repeat lap 1. Data volume isn't the constraint (training takes about 13 windows/s). Episode length is stored per episode, not hard-coded, so longer episodes stay possible.
- Q2 mix → **exactly 1000 episodes per Instruction × Direction**, assigned by episode index (`k mod 6`), not drawn, so every split stays balanced. Equal thirds of Instructions keep the CFG-unconditional branch mix at 50/50 (outer + inner + `any`'s coin), and keep the `outer`/`inner` contrast as large as `any`. Direction is assigned via `StartSpec(direction=…)`; the Step 2 recording rule still applies (`route=instruction` for `outer`/`inner`, random under `any`).
- Q3 start randomisation → **Step 1 `reset` ranges unchanged** (any Station, either Route, lateral U[±1 m], heading U[±10°], speed U[0, 6] m/s); "randomise speed" means the start speed only. Measured over 300 × 300-tick episodes at default noise: lateral > 0.3 m on 3.4% of ticks (> 0.6 m: 1.1%); speed off-profile by > 1 m/s on 3.1%. Recovery coverage is thin, and speed recovery comes only from starts (bursts are steer-only). Widening the ranges would mean re-checking the expert (crash margin about 1 m). **Open point for Step 9:** if the learned policy drifts off line or off speed, prefer DAgger relabelling with the pure `Expert.control` over wider starts.
- Q4 expert params → **Step 2 defaults, fixed for every episode**: `a_lat_plan = 5.0`, `v_cap = 10`, `burst_peak = 0.4`, `burst_gap_s = 3.0`, `p_outer = 0.5`. Style knobs fixed because of the hidden-variable caveat; noise at the swept-safe default (about 24% perturbed ticks, about 55% usable window starts). Params go into the manifest, so a mixed-style dataset is just a different recording later.
- Q5 splits → **by episode, 80/10/10: train 4800, val 600, test 600** (800/100/100 per Instruction × Direction; combination = `k mod 6` within each split). **Seed blocks:** train `1_000_000 + k`, val `2_000_000 + k`, test `3_000_000 + k`; Step 9 closed-loop eval `9_000_000 + k` (never recorded); Step 2 `eval`/pytest keep `0…999`. Blocks never overlap, and a split can grow without shifting the others. Roles: val for model selection in Steps 4–8, test for the Step 8 held-out check and final numbers only.
- Q6 crashes → **the recorder stops with an error** naming seed, Instruction and Direction. A crashed episode is never kept (it would teach a failing demonstration) and never silently dropped or replaced (that breaks the seed ↔ episode mapping and the balance, and biases towards easy starts). A crash means an expert or simulator regression, to be fixed and re-checked with `eval`. The only accepted ending is `termination = "max_steps"` at 300 ticks.

**Resolution (2026-11-03):**

| Split | Episodes | Per Instruction × Direction | Seeds (episode k) |
|---|---|---|---|
| train | 4800 | 800 | `1_000_000 + k` |
| val | 600 | 100 | `2_000_000 + k` |
| test | 600 | 100 | `3_000_000 + k` |
| total | **6000** | 1000 | — |
| *Step 9 closed-loop eval (not recorded)* | — | — | `9_000_000 + k` |
| *Step 2 `eval` / pytest* | — | — | `0 … 999` |

- Combination of episode k within a split: `(INSTRUCTIONS × (cw, ccw))[k mod 6]`, with `StartSpec(direction=…, route=instruction or None for any)`.
- Episode length **300 ticks** (stored per episode); start ranges = Step 1 `reset` defaults; expert = `ExpertParams()` defaults, fixed.
- Size: 6000 × 300 = **1.8M ticks** (1.806M States including reset States), about **0.98M usable clean-future windows** at about 163 per episode (train about 0.78M), about 6900 Passages. Storage about 0.07 GB (states-only format, ticket 01). Recording takes under a minute in parallel.
- Any crash aborts the recording.
- Open point for Step 9: recovery coverage is thin (lateral > 0.3 m on 3.4% of ticks); prefer DAgger over wider starts if needed.
