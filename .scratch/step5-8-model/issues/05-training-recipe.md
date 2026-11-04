# Training recipe

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-throughput-budget.md, 03-transformer-blocks.md, 04-flow-matching-objective.md
Parent: ../MAP.md

## Question

How is the model trained, for how long, and when is a run good enough to freeze?

- Data: uniform over train Windows (1.29 M), or oversample `approaches_section` / `any` Windows? Probes never. Colour jitter is already in the data.
- Instruction dropout for classifier-free guidance (PLAN Step 10: ~10%) — in every run, or a switch?
- Optimiser, lr, warm-up, schedule, batch, steps (within ticket 01's hours), weight decay, gradient clipping, EMA (FLUX keeps EMA copies)?
- Monitoring: train loss per stream (PLAN Step 7 check), val loss per stream on a fixed `eval_subset`, periodic sampling metrics (ticket 07) — how often?
- Seeds and determinism, resume (reuse the Step 4 pattern), Run directory layout `runs/wam/…`.
- Stopping / freezing rule and what a failed run triggers (Step 4: reopen design, no seed rerolls).

Check: a recipe table (data, optimiser, schedule, batch, steps, expected hours), monitoring list, and a freeze rule.

## Comments
