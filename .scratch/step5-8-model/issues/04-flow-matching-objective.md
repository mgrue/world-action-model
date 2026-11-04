# Flow-matching objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
Blocked by: 02-tokens-and-positions.md
Parent: ../MAP.md

## Question

Exactly what is noised, at which level, and what is the loss?

- PLAN: `x_t = t·noise + (1 − t)·clean`, target `noise − clean`, MSE. One t per Window shared by future Frames and actions (PLAN), or independent t per stream (lets Step 9 condition on clean frames, FLUX-style)?
- Distribution of t: uniform, logit-normal (SD3 / FLUX), shifted?
- Scale of each stream: Latents are unit-variance per channel (vae-v1); actions are in [−1, 1] with their own spread (steer small, throttle often saturated?) — normalise actions per dimension from train stats?
- Loss weighting: action weight w_a vs video (PLAN: weight actions up; ≈ 1024 video values vs 32 action values per Window); per-token mean vs sum.
- Step 10 switches: no video loss (drop future Frame tokens entirely, or keep them noised but unweighted?), regression baseline objective.
- Measure: action-label statistics on train (mean/std per dimension, how bimodal at the Fork for `any`).

Check: the exact loss formula with all weights and the t distribution, action normalisation rule, the Step 10 variants written as config switches, and the action statistics that justify them.

## Comments
