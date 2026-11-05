# Replanning scheme

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
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

## Resolution

**Default closed-loop scheme** (grilled with mgrue, 2026-11-05): the best training-free scheme we can justify, so Step 9 judges the model, not a naive loop.

| Parameter | Default | Exposed for sweeps |
|---|---|---|
| `execute_k` (actions executed per replan) | **1** | 1, 2, 4, 8, 16 |
| `seed_mode` | **per_replan**: initial noise per episode row from generator seeded by (episode seed, Tick), independent of batching | `per_episode` (same noise every replan; measured worse, kept as an experiment) |
| `n_samples`, `select` | **4, `coherence`** with `rho = 0.9`: keep the candidate with the smallest Σ_τ ρ^τ ‖a_new[τ] − a_prev[k+τ]‖ over the 16 − k overlapping steps of the previously *selected* Chunk; first replan of an episode takes sample 0 | `first`; any N |
| `n_steps` | **4** | 1–20 |
| `cfg_action` / `cfg_video` | **1 / 1** (off) | any |
| Ensembling / averaging of Chunks or samples | **not offered** | — |

Evidence (`prototypes/replan_variants.py`, 24 dev episodes, seeds `CLOSED_LOOP_SEED_BLOCK + 1000 … 1023`; flips = Passage approaches whose replans 1–12 Ticks before entry classify to different Branches against the Steps 5–8 Fork references):

| Variant | Crash | Laps | Wrong Passages | `any` flips | Wall (24 eps, MPS) |
|---|---|---|---|---|---|
| execute 1 | 21 % | 3.3 | 3 | 31/31 | 6.4 min |
| execute 1, 4 steps | 21 % | 3.2 | 2 | 32/32 | 2.7 min |
| execute 1, CFG 2 | 12 % | 3.5 | 1 | 32/32 | 12.5 min |
| execute 1, per-episode noise | 33 % | 2.8 | 1 | 7/31 | 5.8 min |
| execute 1, ACT ensembling (m = 0.01) | 96 % | 0.4 | 0 | 5/5 | 2.2 min |
| execute 2 | 42 % | 2.8 | 3 | 28/30 | 3.0 min |
| execute 2, per-episode noise | 62 % | 1.8 | 1 | 10/26 | 2.2 min |
| execute 2, coherence best of 8 | 46 % | 3.0 | 0 | 3/26 | 20 min |
| execute 4 | 88 % | 1.4 | 0 | 10/13 | 1.0 min |
| execute 4, coherence best of 8 | 83 % | 1.3 | 0 | 1/12 | 5.4 min |
| **execute 1, coherence best of 4, 4 steps (default)** | 25 % | 3.1 | 1 | 14/32 | 9.3 min |
| execute 1, coherence best of 4, 4 steps, CFG 2 | 33 % | 2.9 | 0 | 16/34 | 16.6 min |

Reading:
- Crashes are set by `execute_k` (21 → 42 → 88 % at 1 / 2 / 4): replanning is the model's only feedback against lateral drift (see [Does wam-v1 drive at all?](01-does-it-drive.md)). Nothing else loop-side moves them beyond noise (24 episodes: 3–8 crashes is within noise); CFG 2's 12 % did not replicate.
- Fork flips and wrong Passages are set by sample selection: coherence halves flips at N = 4 / execute 1, nearly removes them at N = 8 / execute 2. Fresh-noise flips never caused an Island Crash; they show up as occasional wrong Passages.
- Per-episode fixed noise reduces flips but adds crashes (a bad draw persists all episode). ACT ensembling delays every correction → 96 % crashes.
- 4 Euler steps are as good as 10 at 2.3× less cost.
- Best of 8 would cut flips further at ≈ 2× cost; left to [Metrics and gates](05-metrics-and-gates.md) to decide whether flips matter enough.
- Defaults were picked on dev seeds; the gate set must be disjoint ([Episode protocol](04-episode-protocol.md)).
