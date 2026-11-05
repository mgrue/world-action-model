# Replanning scheme

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
Blocked by: 01-does-it-drive.md, 02-chunk-execution-practice.md
Parent: ../MAP.md

## Question

How does the driver turn Chunks into executed actions?

- Replan interval (execute k of 16), synchronous at B = 1 vs lockstep-batched episodes.
- Samples per replan and selection rule (first sample vs something smarter), seeding per replan (deterministic from episode seed + Tick).
- `n_steps`, `cfg_action` / `cfg_video` defaults for closed loop.
- Fork consistency: does commitment hold across replans under `any`, or does it flip-flop into the Island? Whether a consistency rule (ensembling, keeping the previous Chunk's branch) belongs in the driver.

Check: one default scheme with its measured closed-loop numbers vs the alternatives tried (prototype on the closed-loop seeds' dev subset), and the list of scheme parameters the driver exposes.
