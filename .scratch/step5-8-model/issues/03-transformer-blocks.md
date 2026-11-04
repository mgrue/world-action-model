# Transformer blocks and noise conditioning

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
Assignee: mgrue
Blocked by: 01-throughput-budget.md, 02-tokens-and-positions.md
Parent: ../MAP.md

## Question

What does one block look like, and how big is the stack?

- FLUX single-stream block: parallel attention + MLP (SwiGLU), QK RMSNorm, adaptive layer norm modulation (shift, scale, gate) from the noise level. Mirror it, or plain sequential pre-norm blocks with adaLN-Zero (DiT)?
- Noise level per token: clean tokens (obs, State, Instruction) get t = 0 and noised tokens the sample's t (FLUX "per-token noise level")? Timestep embedding (sinusoidal + MLP).
- One shared stack (PLAN), no per-stream private blocks for now; width, depth, heads, MLP ratio within ticket 01's envelope (5–20 M params per PLAN).
- Initialisation (zero-init gates and output heads), dropout (none?).
- Which Step 10 switch lives in the architecture: the regression baseline (same trunk, direct action head, no noise?).

Check: a block diagram / layer table, parameter count, measured ms/step at the chosen size, and a short prototype showing the loss falls on a few hundred train Windows.

## Comments
- From 02: input projections/heads and the 162-token layout are fixed (see 02 Resolution); 3-axis RoPE (t 32 / h 16 / w 16 per 64-dim head); modulation vector = per-token noise level only (Instruction is a token, not in AdaLN). Open here: final-layer form (AdaLN-out as FLUX).
- From 04: regression baseline uses this same model: zero action-token inputs, noise-level modulation fixed at t = 1, action head reads clean normalised actions (no extra head). With `video_stream = false` the sequence is 34 tokens (no future-Frame projection/head used); `horizon` changes the token count (162 / 90 / 54).
- Grilled 2026-11-04 (Q1–Q4), reference flux-action `models/transformer.py` (`SingleStreamBlock`/`ModeBlock`, `Modulation`, `LastLayer`, `JointSingleSeq`: one Modulation per stream per stage, shared by all blocks of the stage; per-token timestep vec).
- Q1 block: mirror FLUX's parallel single-stream block — `h = (1+scale)·LN(x)+shift` (LN no affine), `qkv = W h`, RMSNorm on q and k, RoPE, SDPA; `out = W_o·attn + W_down·(SiLU(W_g h) ⊙ W_u h)` (SwiGLU ratio 3); `x += gate·out`; no biases in the block.
- Q2 noise conditioning: per-token `vec = MLP(sinusoid(1000·t, 256))` (256 → d → d, SiLU, no bias), t = 0 for obs / State / Instruction, the Window's t for future Frame and action (t = 1 for regression actions). One `Modulation` = `Linear(d → 3d, no bias)` on SiLU(vec) **per Stream** (obs, state, instr, fut, act), shared by all blocks (≈ 1 M params; per-block would be ≈ 8 M). FLUX `LastLayer` per predicted Stream: `Linear(LN(x)·(1+scale)+shift)`, own adaLN `Linear(d → 2d)` — settles 02's AdaLN-out question.
- Q3 size: d = 256, 8 blocks, 4 heads × 64 (head size fixed by 02's RoPE split), SwiGLU ratio 3; config fields. Escalation for 05: d384×8 if the 100k-Window quick run underfits (train ≈ val loss, both plateaued).
- Q4 init: adaLN-Zero — zero all 5 Modulation linears (blocks start as identity) and each LastLayer's adaLN and output linear (initial velocity 0, first loss ≈ 2); xavier_uniform for other linears, Instruction Embedding N(0, 0.02), timestep MLP N(0, 0.02). No dropout (≈ 1 pass over 797k Windows); weight decay → 05.
- Prototype `prototypes/model_proto.py` (decisions of 02–04 in one file; `uv run --with einops`; out: `out/model_proto.json`, `model_proto_loss.csv`, `model_proto_stdout.txt`):
  - Params **8.23 M**: blocks 6.82 M, Modulations 0.98 M, LastLayers 0.28 M, timestep MLP 0.13 M, input projections 0.02 M.
  - Token counts check: 162 default, 290 stride 1, 90 / 54 at H = 8 / 4, 34 without video. All switch combinations run forward + backward; zero-init heads output exactly 0; regression path (zero inputs, t = 1) runs.
  - MPS, batch 32, fp32, SDPA: **236 ms/step = 135 Windows/s model-only**, ≈ 85 W/s end to end with the 4.4 ms/Window vae-v1 encode (ticket 01 predicted 88) → ≈ 3.3 h per 1 M Windows.
  - Overfit 512 real train Windows, 1500 steps, AdamW 3e-4, grad clip 1.0, `p_uncond` 0.1: mean of first → last 100 steps `loss_video` 1.66 → 0.22, `loss_action` 1.20 → 0.045 (hi t 1.36 → 0.030, lo t 1.10 → 0.053). Both Streams learn; no instability.

**Resolution (2026-11-04):** one shared stack of 8 FLUX parallel single-stream blocks at d256 (4 heads × 64, SwiGLU ×3, QK RMSNorm, no biases, SDPA, 3-axis RoPE), per-token timestep embedding → 5 per-Stream Modulations shared by all blocks, FLUX LastLayer per predicted Stream (future Frame d→64, action d→2), adaLN-Zero init with zero output heads, no dropout. 8.23 M params, 236 ms/step at batch 32 on MPS (≈ 85 W/s end to end). Regression baseline = same model (ticket 04). d384×8 is the documented escalation.

Layer table (d = 256):

| Part | Shape | Params |
|---|---|---|
| frame_in (obs + fut, shared) | Linear(64 → d) + bias | 16.6 k |
| state_in / act_in | Linear(5 → d), Linear(2 → d) + bias | 2.3 k |
| instr | Embedding(4, d) | 1 k |
| t_in | Linear(256 → d), SiLU, Linear(d → d), no bias | 131 k |
| mods × 5 | Linear(d → 3d), no bias | 983 k |
| block × 8 | qkv d→3d, mlp_in d→2·768, attn_out d→d, mlp_out 768→d, RMSNorm q/k | 852 k each, 6.82 M |
| fut_out / act_out | LN, ada Linear(d → 2d), Linear(d → 64 / 2) | 279 k |

