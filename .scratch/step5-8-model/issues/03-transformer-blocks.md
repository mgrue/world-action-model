# Transformer blocks and noise conditioning

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
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
