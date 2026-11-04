# Training recipe

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 01-throughput-budget.md, 03-transformer-blocks.md, 04-flow-matching-objective.md
Parent: ../MAP.md

## Question

How is the model trained, for how long, and when is a run good enough to freeze?

- Data: uniform over train Windows (797k), or oversample `approaches_section` / `any` Windows? Probes never. Colour jitter is already in the data.
- Instruction dropout for classifier-free guidance (PLAN Step 10: ~10%) — in every run, or a switch?
- Optimiser, lr, warm-up, schedule, batch, steps (within ticket 01's hours), weight decay, gradient clipping, EMA (FLUX keeps EMA copies)?
- Monitoring: train loss per stream (PLAN Step 7 check), val loss per stream on a fixed `eval_subset`, periodic sampling metrics (ticket 07) — how often?
- Seeds and determinism, resume (reuse the Step 4 pattern), Run directory layout `runs/wam/…`.
- Stopping / freezing rule and what a failed run triggers (Step 4: reopen design, no seed rerolls).

Check: a recipe table (data, optimiser, schedule, batch, steps, expected hours), monitoring list, and a freeze rule.

## Comments
- From 04: loss and switches fixed (see 04 Resolution). If `loss_action` stalls, sweep `action_loss_weight ∈ {1, 5}` in a 100k-Window quick run. Logs: `loss_video`, `loss_action`, `loss_action_hi_t` (t > 0.7), `loss_action_lo_t`. Action normalisation constants (steer /0.618; throttle −0.077, /0.373) go into the config.
- From 03: model 8.23 M params at d256×8; 236 ms/step at batch 32 on MPS (≈ 85 W/s end to end, ≈ 3.3 h per 1 M Windows). Overfit sanity run used AdamW 3e-4, grad clip 1.0, no warm-up, no weight decay — stable; a starting point, not a decision. Escalate to d384×8 only if the quick run underfits.
