# Linear probe on vae-v1 Latents

Labels: wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What does a linear probe read from frozen `vae-v1` Latents, and how should the probe be defined so that it scores `vae-v1` and `jepa-v1` the same way?

- Probe design: targets (car x, y in metres; heading as sin/cos, decoded with atan2), inputs (flattened normalised grid, 1024 values for `vae-v1`), fit (ridge regression, closed form, regularisation picked on a held-out part), how many train Frames, which eval Frames (the Step 4 check used 4096 val Frames).
- Does a *linear* readout of position from a spatial grid work at all? Position is encoded by *where* activations sit, so a linear map can read it only if the features are localised. If plain ridge falls far short, consider a fixed, still-linear variant (e.g. per-cell readout + softmax-free weighting) and say why.
- Measure on clean val Frames and report per colour Probe; slices as in Step 4 (`all`, `section`, `fast`, `kerb`).

Check: a prototype in `prototypes/` with position p50/p95/max and heading p50/p95/max for `vae-v1`, compared with the Step 4 pose-estimator numbers; a recommended probe definition for tickets 04 and 06.

## Comments
