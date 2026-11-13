# Architecture and objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
Blocked by: 01-linear-probe-on-vae-v1.md, 02-jepa-recipes-at-small-scale.md
Parent: ../MAP.md

## Question

What exactly is `jepa-v1`, and does it reach the probe bar in prototype runs?

- Context encoder: ViT width (~192?), depth (~6?), heads, patch 8 px (64 tokens, grid matches `vae-v1`) vs 4 px (256 tokens, pooled to 8 × 8?), 2D RoPE, normalisation.
- Output feature grid: 8 × 8 × C with C ≈ 32 (a 1×1 projection from width?), so 2 × 2 WAM tokens hold 4C values.
- Target encoder: EMA copy, momentum schedule; target normalisation.
- Predictor: narrow transformer, mask tokens with RoPE positions, depth/width.
- Masking: number and size of hidden rectangles; share that must cover the car (PLAN: at least half), using the car position from the State; context masking.
- Loss: squared error at hidden patches on (normalised) target features; any anti-collapse term.
- Prototype runs on MPS, scored with the probe from ticket 01 and the collapse diagnostics from ticket 02.

Check: one architecture and objective with measured prototype numbers (probe position/heading p95, collapse diagnostics, params, Frames/s, training minutes) vs the alternatives tried.

## Comments
- From 01: score prototypes with the MLP readout from [01](01-linear-probe-on-vae-v1.md) (`prototypes/probe_vae.py`, ≈ 1.5 min); `vae-v1` reference pos p95 0.24 m, head p95 1.7°. Ridge alongside as the explicitness diagnostic.
