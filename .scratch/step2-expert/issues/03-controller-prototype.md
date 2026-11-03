# Expert controller and speed profile

Labels: wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by: 01-controller-research.md
Parent: ../MAP.md

## Question

What exact control law and speed profile let the expert follow either Route in either Direction from every valid start pose, without Crashing, at a sensible pace (not crawling, not on the grip limit)?

- Steering: controller from the research ticket, its gains/lookahead (fixed or speed-scaled), and how it handles the Fork/Merge where both Routes coincide and then split (target Route switches at the section boundary).
- Speed: target speed per Station from curvature (`√(a_lat·R)` with margin), braking horizon using BRAKE = 8 m/s², behaviour from awkward starts (6 m/s already on a branch arc, ±10° heading, ±1 m lateral).
- Smoothness: are the expert's actions smooth enough to be learnable 16 at a time (steer chatter, bang-bang throttle)?

Prototype: a throwaway controller run headless over many seeds × Route × Direction, plus plots/PNGs of trajectories and action traces, and an optional `--show` window for the human to watch.

Check: over ≥ 1000 seeded starts per Route × Direction, zero Crashes in 900-tick episodes, the correct branch on every passage, ≥ 4 laps per episode on average, and the human agrees the driving and action traces look right.

## Comments
- Prototype: [prototypes/expert_prototype.py](../prototypes/expert_prototype.py) (`sweep`, `plot`, `show`) with the research ticket's controller (curvature feed-forward plus lateral/heading feedback, ωs = 0.5) and speed profile (a_lat_plan = 5, brake 6, accel 4), plus ticket 02's Planned-branch rules. Plots are in `prototypes/out/`.
- **Sweep result (2026-11-03):** 1000 seeds × 3 Instructions × 2 Directions, clean and with burst noise (peak 0.4, about 1 s, gaps about 3 s): **0 crashes and 0 wrong-branch passages in 12,000 episodes**, 4.0–4.1 laps per 900-tick episode, `any` split 48% outer (the coin checks fair; cw and ccw share seeds). This took three fixes, all at the ends of the fork section, where the branches coincide:
  1. **Start branch by heading, not nearest distance.** Starting on a branch within about 2.5 m of the Fork or Merge with a lateral offset, nearest-projection named the other branch, and the expert turned 90° into it and crashed. Rule: at reset inside the section, keep the preferred branch (the Instruction, or the coin for `any`) unless its heading error is above **20°** (beyond the ±10° start range) and the other branch fits better. Under `any` this matters because the start route is random and independent of the coin.
  2. **Tie at the Fork:** at s ≈ 0 both branches have identical pose, so any argmin picks "outer". Hence the rule is "keep preferred unless clearly wrong", not argmin.
  3. **Passage scoring uses the branch at the section midpoint** (branch centres 12 m apart), not the exit tick. A passage that never reaches the midpoint (a start past it) is **partial**, so it is exempt: about 2.5% of passages, all first passages.
- **Step 1 gap found:** `Info.branch`/`last_branch` come from nearest-projection `locate`, so on the exit tick they name the wrong branch in about 1.5% of clean passages and **about 27% of passages with noise**. `last_branch` is unusable for instruction scoring as specified. Fix in Step 1 or define scoring in ticket 05 (open question to the human).
- Safety rule (switch plan to the occupied branch when > 2.5 m from the planned Route inside the section) never fired in clean or burst runs; keep it for Step 9.
- Actions: steer is piecewise constant with steps at primitive boundaries (exact feed-forward under instant steer); throttle is deadbeat (full accel out of curves, about −0.75 braking, 0 cruising) with slight ±0.05 chatter while braking. Speeds: 4.9 m/s branch arcs, 6.1 m/s end arcs, 10 m/s briefly on the bottom straight.
