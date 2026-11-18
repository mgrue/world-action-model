# Architecture and objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
Assignee: mgrue
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
- From 02 ([research](../research/02-jepa-recipes-at-small-scale.md)): start from its recommended recipe; PLAN's "at least half of the rectangles cover the car" works against I-JEPA (targets are cut from the context, so a fully hidden car is unpredictable and gets dropped). Car bias should be partial: a target covers some car patches while at least one stays visible. Compare bias off / partial / full-cover.
- **Progress (2026-11-18), not yet resolved.** Prototype [`jepa_proto.py`](../prototypes/jepa_proto.py) (research/02 recipe; flags for car bias, Frame-variance hinge, fixed EMA) and [`readout_ckpt.py`](../prototypes/readout_ckpt.py); outputs in `prototypes/out/jepa_*/` (`log.csv`, `diag.jsonl`, `result.json`). ~500 Frames/s on MPS (2.7 M encoder + 0.45 M predictor), so 30k steps ≈ 2.2 h. Decided with the user: **8 px patches** (8 × 8 grid, as `vae-v1`), **Linear 192 → 32 head with the loss on its LayerNorm'd output** (no PCA), **8k-step screens then one 30k confirmation**. Screens (Pose readout, MLP gate tier):

  | variant | MLP pos / head p95 | ridge pos / head p95 | car contrast end |
  |---|---|---|---|
  | `vae-v1` reference | 0.24 m / 1.7° | 1.45 m / 22° | – |
  | partial bias, 6k | 1.95 m / 88° | 2.40 m / 106° | 1.00 |
  | **car bias off** | **1.07 m / 19.8°** | 1.87 m / 104° | 1.39 (rising) |
  | full-cover bias | 1.85 m / 117° | 2.17 m / 108° | 1.08 |
  | partial + Frame-variance hinge (1.0, γ 0.5) | 3.61 m / 174° | 3.03 m / 124° | 1.01 |
  | partial + fixed EMA 0.9995 | 1.98 m / 174° | 1.49 m / 75° | 1.02 |

  Every variant drops the car in the first ~2k steps (car contrast 2.3 at random init → ≈ 1.0); global std and rank stay healthy, so only the car-contrast and readout diagnostics see it. The Frame-variance hinge is met by episode colour, not the car. Car bias (partial or full) hurts. Colour Probes break every variant (MLP pos p95 11–37 m). 30k confirmation of car bias off follows in the resolution.
- **Resolution (2026-11-18): negative.** The 30k confirmation of car bias off (`jepa_c30k_off`, 125 min, 511 Frames/s) ends at **MLP pos p95 7.39 m, head p95 176°** (ridge 5.54 m / 163°; colour Probes 13–27 m); car contrast sits at 1.00 from step 2k to 30k while effective rank rises to 26/32. The 8k screen's 1.07 m / 19.8° was a snapshot of the short schedule's annealing, not learning; longer training removes the car completely. **No I-JEPA recipe tried yields a `jepa-v1` that sees the car**; the encoder settles on Track position + episode colour, which predicts every background target on a fixed Track.

  Decided and carried forward (with the user): 8 px patches, 8 × 8 grid; ViT 192 wide, 6 deep, 3 heads, 2D RoPE (2.71 M params) with a Linear 192 → 32 head and the loss on its LayerNorm'd output; predictor 96 wide, 4 deep (0.45 M); I-JEPA multi-block masks; the prototype as the starting code. Ruled out: car-biased masking in both forms (PLAN's "at least half the rectangles cover the car" hurts), a position-centred Frame-variance hinge (met by colour), slow fixed EMA alone. For **Check and gate**: global std and effective rank pass a car-blind encoder; car contrast and the Pose readout are the diagnostics that catch it. Destination kept (the user's choice); next: [A JEPA objective that keeps the car](09-jepa-objective-that-keeps-the-car.md).
