# JEPA recipes at small scale

Labels: wayfinder:research
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What do I-JEPA (and the follow-up literature) say about the choices `jepa-v1` must make, and what carries over to 64 × 64 Frames, ~1 M train Frames of one fixed Track and an M1 Pro?

- Masking: number, size and aspect of target blocks; context block; how masks are sampled per batch. Precedent for biasing masks toward an object of interest (our car-covering masks).
- Target encoder: EMA momentum and schedule; whether targets are layer-normalised; which layer's output is the target.
- Predictor: depth/width relative to the encoder; positional information for hidden positions (mask tokens + RoPE).
- Collapse: what prevents it, how it shows up, which diagnostics are standard (per-channel std, rank / effective rank of features, probe accuracy during training). Whether extra regularisers (VICReg-style variance terms) are used or needed at small scale.
- Small scale: any results on small images (CIFAR, 32–96 px), tiny ViTs, short training, few classes; known failure modes and the training length needed.
- Output used downstream: final-layer patch features vs an average of layers; feature dimension; normalisation of features for a downstream generative model.

Check: a findings file `research/02-jepa-recipes-at-small-scale.md` with sources, a recommended starting recipe for ticket 03 (with the numbers to try) and a list of collapse diagnostics for ticket 04.

## Comments

- Resolved (research). Start from I-JEPA's official recipe translated to the 8 × 8 grid: 4 target blocks of 3×3 / 3×4 / 4×3 (scale 0.15–0.2, aspect 0.75–1.5), one context block at scale 0.85–1.0 minus the targets; LayerNorm'd last-layer targets from an EMA encoder (0.996 → 1.0, linear); a predictor narrower than the encoder (96 wide, 4 deep), RoPE mask tokens; no regulariser in the baseline. Two surprises:
  - **Car bias must be partial.** A target that covers the whole car hides it from the context, which teaches the encoder to drop the car. Cover some of the car's patches and leave at least one visible.
  - **A fixed Track allows a position-only degenerate solution.** The collapse checks must measure variation across Frames at a fixed position and car contrast, not only global channel std.
  - Diagnostics and alternatives are listed in [research](../research/02-jepa-recipes-at-small-scale.md).
