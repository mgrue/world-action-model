# Sampler

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
Assignee: mgrue
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
- From 05: the sampler also runs inside training every 8,000 steps (EMA weights, 10 Euler steps, 256 val Windows) — needs a batched, no-grad entry point.
- Grilled 2026-11-04 (Q1–Q3), reference flux-action `inference/sampling.py` (DROID policy: UniPC order 2, 4 steps, shift 5, CFG 3 on video only; `euler` on `timeshift(alpha, linspace(1, 0, n+1))`; `cfg_two_pass` with per-stream scales).
- Q1 integrator: plain Euler, `n_steps = 10` default (argument), schedule `timeshift(t_shift, linspace(1, 0, n_steps+1))` with the Checkpoint's training `t_shift` (1.65); x ← x + (t_next − t)·v for future Latents and actions together. UniPC noted as a later speed-up, not built. Quality vs `n_steps ∈ {2, 4, 10, 20}` is measured on a trained model (07 / Step 10). Regression: one forward at t = 1 with zero action inputs, `n_steps` ignored.
- Q2 CFG: per Stream `v = v_uc + w·(v_c − v_uc)`, v_uc = Instruction `null`; separate `cfg_action`, `cfg_video`, both default 1.0 (off, single pass); when on, one batched forward of 2B (conditional ‖ null). Rejected for `objective = regression`. Suggested Step 10 sweep: w ∈ {1, 1.5, 2, 3}, both / action only / video only.
- Q3 API: `wam.policy.load(name_or_run, device=None) -> Policy` (EMA weights + config; loads `vae-v1` by the name in its card); `Policy.act(frame, state, instruction, *, n_samples=1, n_steps=10, cfg_action=1.0, cfg_video=1.0, seed=None, return_latents=False) -> result` with `actions` np.float32 (B, n_samples, H, 2) raw, clipped to [−1, 1]; `latents` torch (B, n_samples, K, 16, 8, 8) normalised or None (decode with `policy.compressor.decode`); `future_k` (K,). Inputs: uint8 Frame (64,64,3) or (B,…), State features (5,) or (B,5), Instruction name / int or a sequence of B; leading B kept for single inputs; `n_samples` repeats inputs along the batch. Tensor-level `Policy.sample(obs_z, state, instr, *, …)` batched, no-grad — used by `act()` and inside training (05). Seed: int or `torch.Generator`, noise drawn on CPU then moved (bit-identical on the same device).
- Prototype `prototypes/sampler_proto.py` (ticket-03 model, random non-zero weights; out `out/sampler_proto.json`, `sampler_proto_stdout.txt`), MPS, fp32:

  | Model | B | Steps | CFG | ms per call | ms per Chunk |
  |---|---|---|---|---|---|
  | video (162 tok) | 1 | 10 | off | **65.9** | 65.9 |
  | video | 1 | 10 | 2.0 (2B batched) | 87.5 | 87.5 |
  | video | 1 | 4 / 20 | off | 27.0 / 121.7 | |
  | video | 16 | 10 | off | 347 | 21.7 |
  | video | 256 | 10 | off | 5,500 | 21.5 |
  | no video (34 tok) | 1 | 10 | off | 44.9 | 44.9 |
  | no video | 256 | 10 | off | 988 | 3.9 |

  `vae-v1` encode of one Frame 2.6 ms. Same seed → identical Chunk (same device): yes.
  - Step 9 replan ≈ 2.6 + 66 ≈ 70 ms (≈ 18 ms per Tick when replanning every 4 Ticks); CFG adds ≈ 33 % at B = 1 (batched, not 2×).
  - Periodic sampling in training (05: 256 val Windows, every 8,000 steps) ≈ 5.5 s; 07's Fork spread with `n_samples = 16` on 256 Windows ≈ 90 s.

**Resolution (2026-11-04):** Euler, 10 steps on the training-shifted schedule (1.65), per-Stream CFG `cfg_action` / `cfg_video` default off (batched 2B when on), actions de-normalised and clipped; `Policy.act` / `Policy.sample` as above. 66 ms per Chunk at B = 1 on MPS (≈ 70 ms per replan incl. encode), 21.5 ms per Chunk batched.

