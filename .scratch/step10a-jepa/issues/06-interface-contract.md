# Interface contract

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

What do the later WAM switch and the comparison rely on from `jepa-v1`?

- Feature grid: shape (C, 8, 8), dtype, per-channel normalisation (mean/std stored in the Checkpoint), which encoder output (context encoder on the full Frame vs target encoder).
- API: a frame encoder protocol shared with `wam.compressor.Compressor` (`load(name)`, `encode(frames)`, `encode_windows(batch)`, `latent_shape`), with `decode` absent for JEPA; how callers tell the two apart.
- Checkpoint format and card: weights kept (context/target encoder; predictor dropped?), arch, stats, provenance, code hash, check summary.
- The linear probe as a reusable component (fit, save, apply) for both encoders, including on predicted futures later.
- What the WAM will see: 2 × 2 tokens of 4C values; any constraint this puts on C.

Check: the API signatures, the Checkpoint file list and card fields, and a fake-input test showing a `jepa-v1`-shaped grid fits the WAM `patchify` path with only the in/out widths changed.

## Comments
