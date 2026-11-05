# Episode protocol

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-does-it-drive.md
Parent: ../MAP.md

## Question

Which episodes is a driver evaluated on?

- Seeds from `CLOSED_LOOP_SEED_BLOCK` (dev subset for tuning vs held-out gate set), episodes per Instruction × Direction, episode length (900 Ticks?), start rule (Step 2/3 `start_spec`), colour jitter (`sample_render_params`).
- How many episodes give stable estimates (variance from ticket 01), and the time budget per full evaluation.
- Same seeds for every driver (expert and Policies) so rows are paired.

Check: the seed list rule, counts, and expected runtime; confidence intervals for the headline rates at that count.

## Comments
- From 01: per-episode outcomes are very variable at the current skill level (crash ticks 19–822); 60 episodes at execute 4 took 121 s wall, 24 at execute 1 took 420 s (lockstep, MPS). Expert parity holds with the Step 3 start rule + `sample_render_params`.
- From 03: default scheme costs ≈ 9.3 min per 24 episodes (lockstep batched, MPS; ≈ 23 s per episode); plain execute-1 first-sample ≈ 6.4 min. Dev seeds `CLOSED_LOOP_SEED_BLOCK + 1000…1023` were used for tuning (and 0…59 in 01): the gate set must be disjoint. Crash counts at 24 episodes vary 3–8 between near-identical schemes — size the gate set for that.
