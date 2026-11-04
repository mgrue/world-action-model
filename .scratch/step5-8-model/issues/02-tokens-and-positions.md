# Token layout, positions and Instruction encoding

Labels: wayfinder:grilling
Status: closed
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
- Grilled 2026-11-04 (Q1–Q7), reference: flux-action `models/positional.py`, `processing/packing.py` (`pack_actions`, `pack_text`), `models/transformer.py` (`vector_in` fed zeros by `pack_text`).
- Q1 future Frames: 8 of 16, the even ones (k = 2, 4, …, 16 → Step 3 `future_frames=(1, 3, …, 15)`, 0-based). Video is the auxiliary target; one Tick moves ≤ 1 m in a 5 m cell, so neighbours are near-redundant; RoPE keeps exact t alignment; halves model and encode cost for Step 10's many Runs. Training switch `future_stride ∈ {1, 2}`, default 2 (stride 1 → 290 tokens).
- Q2 tokenisation: lossless 2×2 patchify, `rearrange(z, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)` → 16 tokens × 64 values per Frame, row-major 4×4 grid (10 × 10 m each). One `Linear(64→d)` shared by observed and future Frame tokens (they differ by t and noise level); head `d→64` on future tokens only, unpatchified back to (16, 8, 8).
- Q3 Instruction: `nn.Embedding(4, d)`, rows `outer`, `inner`, `any`, `null` (= 3); the row is the token, no projection. `null` ≠ `any` (`any` is data, the coin-flip Fork; `null` = Instruction withheld, never in data). Training switch `p_uncond` (per Window, replace Instruction by `null`), **default 0.1 in every Run incl. baselines** so every Policy can do CFG; `p_uncond = 0` is the control. State is never dropped.
- Q4 conditioning path: Instruction enters **only as a token** (as FLUX Action, whose pooled `vector` is zeros). The AdaLN modulation vector carries only the per-token noise level. Documented fallback if Instruction-following is weak: also add the Instruction embedding to the modulation vector — not built as a switch.
- Q5 positions: 3-axis RoPE (t, h, w), head 64 split t 32 / h 16 / w 16, theta 10000; t in Ticks (obs Frame, State, Instruction at 0; action k and future Frame k at k); h, w = Latent grid cell for Frame tokens, (0, 0) otherwise. No type embedding (per-type projections + t + noise level separate the streams; FLUX has none either).
- Q6 attention: full bidirectional over the whole sequence; no causal mask, no cross-attention, no padding (fixed length). Cost: no caching of the clean tokens across Euler steps — ten 162-token passes per replan, a few tens of ms.
- Q7 layout below. Deferred: final-layer form (AdaLN-out) → 03; action loss weight and action scaling vs Latents → 04; no-video-loss mechanics (drop tokens vs unweighted loss) → 04/07.

**Resolution (2026-11-04):** one Window = **162 tokens** (290 with `future_stride=1`), d256×8 ≈ 88 Windows/s end to end (ticket 01 candidate E).

| # | Stream | Count | Input → d | t | h, w | Clean/noised | Output head |
|---|---|---|---|---|---|---|---|
| 0–15 | observed Frame | 16 | `Linear(64→d)` (shared) | 0 | grid 0..3 | clean | — |
| 16 | State | 1 | `Linear(5→d)` | 0 | 0, 0 | clean | — |
| 17 | Instruction | 1 | `Embedding(4, d)` incl. `null` | 0 | 0, 0 | clean | — |
| 18–145 | future Frame | 8 × 16 = 128 | `Linear(64→d)` (shared) | 2, 4, …, 16 | grid 0..3 | noised | `d→64` → unpatchify |
| 146–161 | action | 16 | `Linear(2→d)` | 1..16 | 0, 0 | noised | `d→2` (velocity) |

Clean = modulated at noise level 0. Concatenation order fixed as above for slicing (irrelevant to the model under RoPE + full attention). Instruction encoding: 4-row embedding with `null`, `p_uncond = 0.1` default. Full attention. CONTEXT.md gains Stream, Token, Chunk, Null Instruction.

