# Training recipe and freezing

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

How is `jepa-v1` trained to completion and frozen?

- Data: `FrameSampler` / `frame_dataset` on the train Split as for `vae-v1`; batch, steps, wall-clock budget on MPS.
- Optimiser, learning-rate and weight-decay schedules; EMA momentum schedule; seeds; resumability.
- Monitoring during training: loss, collapse diagnostics, a cheap probe every N steps.
- Freezing: per-channel feature stats on how many train Frames, then the ticket-04 check, then promote. What happens on a failed check (Step 4 rule: reopen architecture, no seed rerolls?).

Check: a complete recipe with an expected training time, and the freeze-and-promote sequence.

## Comments
