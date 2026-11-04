# Flow-matching objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
Assignee: mgrue
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
- From 02: 128 future-Frame tokens × 64 values = 8192 video values vs 32 action values per Window (`future_stride=2`). Open here: action scaling vs Latents, action loss weight, and no-video-loss mechanics (drop the future tokens → 34-token sequence, or keep them noised with zero weight).
- Measured action-label statistics with `prototypes/action_stats.py` (arrays only; out: `prototypes/out/action_stats.json`, `action_stats_stdout.txt`). Train, 797,262 Windows × 16: steer mean 0.0016, std 0.618, multimodal (peaks ≈ ±0.9, ±0.4, 0 = the curves), 0.1 % at ±1; throttle mean 0.077, std 0.373, ≈ 81 % exactly 0, 10.4 % at +1, ≈ 4 % near −0.8 (braking). Step-to-step change std ≈ 0.15 per Tick (smooth Chunks).
- The Fork with `any` (≈ 20k train Windows approaching the fork section, 50/50 by Planned branch): with steer × Direction sign, the outer Chunk sweeps 0.52 → −0.78, the inner rises 0.57 → 0.82; separation d′ by k = 0.3 (k=1), 1.6 (k=8), 11 (k=16); Chunk-mean steer clearly bimodal. A regression must average here — the case PLAN tests, and H = 16 is long enough to show it.
- Grilled 2026-11-04 (Q1–Q7), reference flux-action `processing/packing.py` (`sample_timesteps`, `add_noise`, `flow_loss`, `ACTION_SCALE`).
- Q1: one shared t per Window for future Frames and actions (PLAN, FLUX Action); clean tokens at t = 0 (ticket 02).
- Q2: t ~ logit-normal(0, 1) then rational shift `t' = s·t / (1 + (s−1)·t)`, s = 1.65 (= e^0.5, FLUX.1's rule at its smallest size; same 2×2-of-16-channel tokens). Shift > 1 tilts training to high noise, where the Branch is decided. Switch `t_shift`, default 1.65; ticket 06 uses the same shift for the step schedule. FLUX Action's shift 42 is for long high-res video, not this.
- Q3: per-dimension standardisation with frozen train constants: `steer_n = steer / 0.618` (mean fixed to 0, cw/ccw symmetric), `throttle_n = (throttle − 0.077) / 0.373`. Unit variance like the Latents, so both streams have comparable signal-to-noise at a shared t. Constants live in the model config / Checkpoint; the Policy API takes and returns raw actions, de-normalising and clipping to [−1, 1].
- Q4: per-stream means, weighted sum (FLUX's action weight 50 in its token-pooled loss is the same idea, ≈ 6× here); no per-t weighting. Switches `video_loss_weight` (w_v) and `action_loss_weight` (w_a), default 1 and 1. Log `loss_video`, `loss_action`, and `loss_action` split at t > 0.7 vs ≤ 0.7.
- Q5: no video loss = switch `video_stream` (default true); false drops the future-Frame stream entirely (34 tokens, no future Frames rendered or encoded). Keeping noised future tokens with w_v = 0 is rejected (low-t future Frames leak the actions; untrained video velocity at sampling) — config rejects `video_stream = true` with w_v = 0. Caveat for 07: the variant also has less compute per action.
- Q6: regression baseline = switch `objective ∈ {flow, regression}`; regression uses the same model with zero action-token inputs at modulation t = 1, the action head predicting clean normalised actions, `L = mse(head, a_n)`, one forward pass at inference. Requires `video_stream = false` (config rejects otherwise); Step 10 compares it with flow + `video_stream = false` (isolates the objective) and full flow as the third point.
- Q7: Chunk length = switch `horizon` H ∈ {4, 8, 16}, default 16: H action tokens at t = 1..H, future Frames at even k ≤ H (stride 2) → 162 / 90 / 54 tokens; Windows from Step 3 `windows(horizon=H)`; normalisation constants unchanged. Caveat for 07: Fork separation grows with k, so short Chunks have less two-answer problem. Replanning rate is inference-only (Step 9 / ticket 08).

**Resolution (2026-11-04):** the objective, per Window (batch mean over Windows):

```
a_n   = ((steer − 0)/0.618, (throttle − 0.077)/0.373)       # (H, 2); Latent tokens z already unit variance
u     ~ N(0, 1);  t = s·σ(u) / (1 + (s−1)·σ(u)),  s = t_shift = 1.65     # one t per Window
ε_z, ε_a ~ N(0, I)  (same shape as the clean future-Frame tokens z (H/2·16, 64) and a_n (H, 2))
x_z = t·ε_z + (1−t)·z ,   x_a = t·ε_a + (1−t)·a_n               # clean tokens (obs, State, Instruction) at t = 0
Instruction → null with prob p_uncond = 0.1                     (ticket 02)
v_z, v_a = model(obs, State, Instruction, x_z, x_a, t)
L = w_v · mean((v_z − (ε_z − z))²) + w_a · mean((v_a − (ε_a − a_n))²)      w_v = w_a = 1
```

Training switches (validated together): `objective` ∈ {flow, regression} (regression: zero action inputs at t = 1, target a_n, needs `video_stream = false`), `video_stream` (default true; false drops the future-Frame stream), `video_loss_weight` / `action_loss_weight` (1 / 1; w_v = 0 only with `video_stream = false`), `t_shift` (1.65), `horizon` ∈ {4, 8, 16} (16), plus ticket 02's `future_stride` (2) and `p_uncond` (0.1). Logged: `loss_video`, `loss_action`, `loss_action_hi_t`, `loss_action_lo_t`.

