# Steps 5–8 world action model spec

Labels: wayfinder:map
Status: open
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step5-8-model/SPEC.md`: a buildable spec for PLAN.md Steps 5–8 (Instruction encoding, the multi-stream transformer, flow-matching training and the Euler sampler) that an agent can implement without follow-up questions — **including the interface contract** Steps 9–10 rely on: how a trained policy is loaded and asked for an action chunk from one Frame, State and Instruction, which training-time switches Step 10's experiments need (regression baseline, no video loss, classifier-free guidance, chunk length), and the model Checkpoint format.

PLAN.md checks to satisfy, turned into measurable checks:
- Step 7: "training loss falls for both streams; logged separately."
- Step 8: "sampled action chunks look like the expert's on held-out windows."

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `CONVERSATION_SUMMARY.md`, explainer `docs/step4-vae.md`). Builds on the implemented Steps 1–4 (`.scratch/step{1..4}-*/SPEC.md`).
- Reference implementation: `/Users/grue/code/flux-action` (`src/flux_action/models/transformer.py`, `processing/packing.py`, `inference/sampling.py`, `models/positional.py`). Read it; mirror its ideas, shrunk; don't import it.
- Vocabulary follows repo-root `CONTEXT.md`; new terms (Token, Stream, Chunk, Policy, …) are added there as tickets resolve.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` where a question needs measuring, research subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research and throwaway prototypes) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no model code in `src/` beyond throwaway prototypes.
- Facts that constrain the model:
  - Frozen compressor `vae-v1` (`wam.compressor.load`): a Frame → normalised Latent (16, 8, 8), 64 row-major tokens of 16 channels, each a 5 × 5 m patch; ≈ 6 ms per Window on MPS, encoded on the fly. The decoder tolerates normalised noise 0.1. Weak spots: off-Route Frames, colour Probes.
  - Windows (Step 3): obs Frame + State features (x/40, y/40, cos θ, sin θ, v/10) + Instruction (`outer` / `inner` / `any`) + H = 16 clean expert actions [steer, throttle] ∈ [−1, 1] + 16 future Frames (action k ↔ future Frame k). `future_frames` can select a subset of the 16 Frames. Train 797,262 Windows with the default clean-future filter (val 98,579, test 98,904); `eval_subset(n)` gives fixed val subsets; `approaches_section` flags Windows before the fork section.
  - The two-valid-answers case PLAN cares about: Instruction `any` at the Fork (expert flips a fair coin per Passage). That is where flow matching should commit and regression should average into the Island.
  - **Throughput is the binding constraint.** Step 3 measured a stand-in 6.3 M-param transformer (d256 × 8) on MPS at full PLAN size (64 × 17 + 1 + 1 + 16 = 1106 tokens): 13 Windows/s, batch 8 max (batch 16 swaps, 32 OOM); d384 swapped. Plain attention. Step 6 needs fewer tokens per Window, memory-efficient attention, or shorter futures.
  - Hardware: Apple M1 Pro, 16 GB, MPS, float32 eager was right for the VAE (ticket 01 of Step 4); Python 3.14, torch 2.15.
  - Step 4 conventions to reuse: `src/wam/` by concern, Runs under `runs/<kind>/<date>-<name>/` (config/log.csv/metrics.json/PNG/stdout, resumable), promoted committed Checkpoints with `card.json` provenance, load by name, CPU tests with a tiny arch < 60 s.

## Decisions so far

<!-- one line per closed ticket -->

- [Throughput and size budget on MPS](issues/01-throughput-budget.md) — SDPA = plain attention on MPS (no memory saving), bf16 20–30% slower, speed ∝ params × tokens; PLAN's 1106-token Window = 15 W/s (19 h per 1 M Windows); envelope: 2×2 patchify (16 tokens per Frame) → 160–300 tokens, 8–20 M params, batch 32, fp32; 290 tokens at d256×8 = 47 W/s end to end incl. 8 ms vae-v1 encode; full run 3–8 h
- [Token layout, positions and Instruction encoding](issues/02-tokens-and-positions.md) — 162 tokens: obs Frame 16 + State + Instruction + 8 even future Frames × 16 (2×2 patchify, shared `Linear(64→d)`) + 16 actions; 3-axis RoPE (t in Ticks, h, w), no type embedding; full attention; Instruction token only via `Embedding(4,d)` incl. `null`, `p_uncond = 0.1` default; switch `future_stride` (default 2)

## Not yet specified

_(empty — every known question has a ticket)_

## Out of scope

- Step 9 closed-loop evaluation and Step 10 experiments themselves — only the API and training switches they need are in scope.
- Changing the frozen compressor (a better one would be `vae-v2` and a new map).
- Free-form text instructions / pretrained text encoders (PLAN Step 5 "optional later").
- Multi-GPU, cloud training, mixed precision unless ticket 01 shows a clear MPS gain.
- Implementing the model (that's what the spec hands off).
