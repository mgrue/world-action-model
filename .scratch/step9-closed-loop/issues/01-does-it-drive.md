# Does wam-v1 drive at all?

Labels: wayfinder:prototype
Status: closed
Assignee: mgrue
Blocked by: —
Parent: ../MAP.md

## Question

Before any design decision: what happens when `wam-v1` drives closed-loop with the most naive loop?

- Throwaway `prototypes/naive_loop.py`: render Frame → `policy.act(frame, state.features(), instruction, n_samples=1, seed=…)` → execute the first 4 actions of the Chunk → replan; 900-Tick episodes from `CLOSED_LOOP_SEED_BLOCK` with the Step 3 start rule and colour jitter; ≈ 10 episodes per Instruction × Direction (≈ 60).
- Measure with `score_episode`/`summarize`: crash rate, laps, Instruction followed, `any` outer fraction; plus where crashes happen (Station, inside/outside the fork section, off-Route), lateral offset, and replan time / wall-clock per episode (B = 1 and episodes batched in lockstep).
- Sanity check of the loop: the expert driven through the same loop structure scores exactly as Step 2 `run_episode` on the same seeds.
- Quick sensitivity: replan every 1 / 4 / 8 / 16 Ticks on a subset.
- Look at it: a few episodes in the pygame window or as a strip of Frames, at the Fork with `any`.

Check: a results table (per Instruction × Direction and per replan interval) + timings in `prototypes/out/`, an expert parity line, and a short written read of the failure modes, recorded on this ticket.

## Resolution

`wam-v1` drives, but **does not hold the line**: with the naive loop it crashes in most episodes, almost never because of the Fork.

- **Loop is correct:** the expert through the same loop structure scores identically to Step 2 `run_episode` (60/60 episodes).
- **Naive loop** (execute 4 of 16, n_samples = 1, 60 episodes, seeds `CLOSED_LOOP_SEED_BLOCK + 0…59`, Step 3 start rule and colour jitter): crash 82 % (expert 0 %), 1.33 laps per episode (expert 4.0), Instruction followed 98 % (1 wrong Passage), `any` outer share 22 % (expert 48 %), mean distance off the centreline 0.43 m (p95 1.06 m).
- **Replan interval** (24 episodes): execute 16 → 100 % crash, 8 → 96 %, 4 → 79 %, **1 → 29 %** (3.2 laps). Replanning is the only feedback the model gets; within a Chunk there is none.
- **Where:** 42 of 49 Crashes outside the fork section, clustered at the end of the top straight / entry of the following end arc (10 m/s, the fastest part). Branch following and commitment at the Fork look fine.
- **Why** (`prototypes/diag_crash.py`: executed action vs `Expert.control` in the model's own State): the model reproduces the *nominal* steer for the place (≈ 0 on the straight, ≈ 0.5 in the end arc) but barely reacts to lateral / heading error; at 0.3–0.6 m off line the expert would steer +0.35 on the straight and 0.86 in the arc. Steer RMSE vs the expert on visited States 0.27 (< 0.3 m off) → 0.57 (0.6–1 m), against 0.064 open-loop on recorded Windows. 0.5 m is under one pixel of the 64 × 64 Frame and the expert data is almost always on the line → classic compounding error of imitation learning.
- **Timing (MPS):** `act` 64 ms per call at B = 1; batched 24.6 / 20.8 / 19.0 ms per Chunk at B = 8 / 24 / 60; render 0.11 ms. Lockstep-batched 24 episodes: 7 min at k = 1, 67 s at k = 4.
- **Decision (grilled with mgrue, 2026-11-05):** Step 9 stays **evaluation only**. Gates are set from what a good driver must achieve, not from what `wam-v1` reaches; `wam-v1` may fail them, and that is a finding (the cost of imitation, PLAN Step 11). Loop-side settings are still chosen by the replanning ticket. Teaching the model to recover (stronger demonstration noise / DART, DAgger, then `wam-v2`) is a separate follow-up map.

Assets: [`prototypes/naive_loop.py`](../prototypes/naive_loop.py), [`prototypes/diag_crash.py`](../prototypes/diag_crash.py); outputs in [`prototypes/out/`](../prototypes/out/) (`naive_main.json` + `_stdout.txt`, `naive_sweep.json` + `_stdout.txt`, `naive_timing.json`, `diag_crash.json` + `_stdout.txt`, `naive_trajectories.png`).
