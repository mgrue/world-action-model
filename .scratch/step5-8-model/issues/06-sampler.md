# Sampler

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
Blocked by: 02-tokens-and-positions.md, 04-flow-matching-objective.md
Parent: ../MAP.md

## Question

How is an action chunk (and the imagined future) sampled from a trained model?

- Euler from t = 1 (noise) to t = 0, number of steps (PLAN ≈ 10; FLUX DROID uses 4), step schedule (uniform vs shifted, `flux_action/inference/sampling.py`).
- Classifier-free guidance: formula, guidance scale, applied to video, actions or both (FLUX DROID: video only) — implemented now, default off?
- Seeding: one generator per call, determinism on MPS, sampling several chunks at once (batch) to see the spread at the Fork.
- Outputs: un-normalised actions (16 × 2, clipped to [−1, 1]), optionally predicted future Latents → `decode` → Frames for inspection.
- Cost per chunk on MPS (Step 9 calls it every few ticks).

Check: sampler pseudo-code with defaults, measured ms per chunk on MPS, and the API signature Step 9 will call.

## Comments
- From 04: velocity target is ε − clean (x_t = t·ε + (1−t)·clean), so Euler steps go from t = 1 to 0 with x ← x + (t_next − t)·v. Step schedule uses the same rational shift `t_shift` (1.65) as training. Output actions are normalised: de-normalise (steer ×0.618; throttle ×0.373 + 0.077) and clip to [−1, 1]. Regression objective: one forward pass, no sampling.
- From 03: one forward pass = full 162-token sequence (no KV caching; clean tokens attend to noised ones). At batch 1, 10 Euler steps ≈ 10 forwards of the 8.2 M model.
