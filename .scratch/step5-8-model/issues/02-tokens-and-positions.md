# Token layout, positions and Instruction encoding

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 01-throughput-budget.md
Parent: ../MAP.md

## Question

What is the token sequence for one Window, and how does each token know what and when it is?

- Streams (PLAN Step 6): observed-Frame Latent tokens (clean), one State token (clean), one Instruction token, future-Frame Latent tokens (noised), one action token per future step (noised). Which future Frames: all 16, or a subset (ticket 01's budget)? Actions always all 16?
- Tokens per Frame: 64 Latent cells as-is, or patchified / pooled (FLUX patchifies 2×2)?
- Step 5: one learned embedding per Instruction; plus a "null" Instruction for classifier-free guidance (Step 10) — trained in from the start?
- Positions: FLUX uses 4-axis RoPE (time, height, width, sequence). Here: time axis so action k and future Frame k line up (observed Frame = time 0, action k / Frame k = time k + 1?), 2D grid position for Latent tokens, type embedding per stream? RoPE vs learned absolute embeddings.
- Per-type input projections and output heads (PLAN).
- Attention mask: full attention over the whole sequence (FLUX: no cross-attention, no causal mask)?

Check: a table of every token in the sequence (stream, count, input → d projection, positions, clean/noised, output head), the Instruction encoding incl. null token, and the total token count within ticket 01's budget. Add Stream / Token / Chunk to `CONTEXT.md`.

## Comments
