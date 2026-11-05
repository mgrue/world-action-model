# Replanning scheme

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by: 01-does-it-drive.md, 02-chunk-execution-practice.md
Parent: ../MAP.md

## Question

How does the driver turn Chunks into executed actions?

- Replan interval (execute k of 16), synchronous at B = 1 vs lockstep-batched episodes.
- Samples per replan and selection rule (first sample vs something smarter), seeding per replan (deterministic from episode seed + Tick).
- `n_steps`, `cfg_action` / `cfg_video` defaults for closed loop.
- Fork consistency: does commitment hold across replans under `any`, or does it flip-flop into the Island? Whether a consistency rule (ensembling, keeping the previous Chunk's branch) belongs in the driver.

Check: one default scheme with its measured closed-loop numbers vs the alternatives tried (prototype on the closed-loop seeds' dev subset), and the list of scheme parameters the driver exposes.

## Comments
- From 02 ([research](../research/02-chunk-execution-practice.md) §3), prototype in this order, all training-free:
  1. Execute-k sweep with plain receding horizon and the first sample: k ∈ {16, 8, 4} (optionally 2). Expected default: k = 8.
  2. Seed mode: the same `seed` every replan within an episode vs a fresh seed per replan. With a fixed seed `flow.initial_noise` is identical every time, which should keep the `any` branch.
  3. BID-lite selection: n_samples = 8, keep the sample with the smallest ρ-weighted L2 to the overlapping tail of the previously executed Chunk (ρ ≈ 0.9), with no forward contrast.
  4. Optional negative control: ACT temporal ensembling / EMA. Expect it to average the branches into the Island.
  5. Only if flip-flops remain: hard-prefix inpainting in a custom Euler loop (`prototypes/`).
- Never average samples or Chunks. Measure the flip-flop rate on the approach to the Fork under `any`.
- Driver parameters to expose: `execute_k`, `n_samples`, `select` (first|coherence) with `rho`, `seed_mode` (per_replan|per_episode), `n_steps`, `cfg_action`, `cfg_video`.
- From 01: execute-k dominates — crash 100 / 96 / 79 / 29 % at k = 16 / 8 / 4 / 1 (24 episodes); the Fork is not where it fails, lateral drift on the top straight → end arc is. Expect k = 1 (or 2) as default; test whether best-of-N selection (and fixed per-episode seed) changes crashes, not just Fork flips. Cost at k = 1: 900 batched calls per episode set (≈ 7 min for 24 episodes lockstep). Results are reported, not tuned to pass: Step 9 is evaluation only.
- Progress (2026-11-05, session paused): prototype `prototypes/replan_variants.py` (24 dev episodes, seeds CLOSED_LOOP_SEED_BLOCK + 1000…1023). Done: k4 (crash 0.88, any flips 10/13), k1 (crash 0.21, any flips 31/31 but 0 Island crashes, 3 wrong Passages), k1_ep (crash 0.33, fork flip line not captured). Results accumulate in `prototypes/out/replan_variants.json`; resume with: `uv run python .scratch/step9-closed-loop/prototypes/replan_variants.py k2 k2_ep k2_coh8 k1_ens k1_steps4 k1_cfg2 k4_coh8` (≈ 1 h).
