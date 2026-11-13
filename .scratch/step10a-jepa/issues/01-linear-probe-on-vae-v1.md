# Linear probe on vae-v1 Latents

Labels: wayfinder:prototype
Status: closed
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
- **Resolution (2026-11-13).** Prototypes: [`probe_vae.py`](../prototypes/probe_vae.py) (ridge + MLP) and [`probe_local.py`](../prototypes/probe_local.py) (local softmax-over-cells probe); outputs in `prototypes/out/probe_*.json`. On `vae-v1`, 65,536 train Frames, the Step 4 eval Frames (4096 val, `carcheck.eval_frames`), errors vs the true State:

  | readout | pos p50 / p95 / max | head p50 / p95 / max |
  |---|---|---|
  | mean (floor) | 12.3 / 16.5 / 17.3 m | 88 / 163 / 180° |
  | ridge, flattened grid (λ 0.1) | 0.54 / 1.45 / 3.2 m | 6.6 / 22 / 146° |
  | local, 3×3 neighbourhood, softmax over cells (725 params) | 0.42 / 1.05 / 2.3 m | 5.3 / 19 / 62° |
  | MLP 1024→512→512→4 | 0.09 / 0.24 / 2.7 m | 0.4 / 1.7 / 15° |

  Colour Probes break every readout (MLP pos p95 7–11 m, `no_kerb` 1.2 m; ridge 6–16 m). Ridge train-fit ≈ held-out, so it is capacity-limited, not data-limited: the grid says *where* by which cell is active, which one linear map can't turn into metres finer than the 5 m cell.

  **Decision (with the user):** rename to **Pose readout** (`CONTEXT.md`; "probe" clashes with Probe). Two tiers, fitted identically on both encoders:
  - **MLP readout gates** (the car's pose is *in* the features, which is what the WAM transformer needs): input = the full normalised feature grid flattened (C·64 values, no pooling), standardised per input on the readout's train Frames; targets (x, y, cos θ, sin θ) standardised; MLP d→512→512→4, GELU; AdamW lr 1e-3, wd 1e-4, batch 256, 4000 steps, cosine to 0, seed 0, CPU float32; heading = atan2(sin, cos). `vae-v1` clears the Step 4 p95 bar with it (0.24 m, 1.7°), not the max limits (2.7 m, 15°).
  - **Linear readout reported**: ridge on the same standardised inputs, bias unpenalised, λ from {1e-2 … 1e4} on a 20 % episode-held-out part of the train Frames (score pos p95/0.3 + head p95/10), then refit on all.
  - Frames: train = 65,536 train-Split Frames, per-episode quotas, ticks from stream `[seed, 7]`; eval = the Step 4 check's 4096 val Frames (stream `[seed, 5]`), clean + each colour Probe (readout fitted on clean train Frames only); Step 4 slices `all/section/fast/kerb`. ≈ 1.5 min per encoder on the Mac.
  - Left to **Check and gate**: absolute vs relative thresholds, whether max limits apply, seed variance of the MLP readout.
