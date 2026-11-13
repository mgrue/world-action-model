# JEPA recipes at small scale

Labels: wayfinder:research
Status: open
Assignee:
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
