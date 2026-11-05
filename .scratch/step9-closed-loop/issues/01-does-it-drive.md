# Does wam-v1 drive at all?

Labels: wayfinder:prototype
Status: open
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
