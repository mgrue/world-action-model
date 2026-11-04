# Architecture and training objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by: 01-torch-on-mac.md, 04-car-visibility-check.md
Parent: ../MAP.md

## Question

What network and loss turn a 64×64 Frame into 8×8×16 latents that keep the car, and give latents that suit flow matching?

- Plain autoencoder vs VAE (KL term and its weight): PLAN says "toy video VAE"; FLUX uses a KL-regularised VAE. What does the KL buy us here (smooth, roughly Gaussian latents for Step 7) and what does it cost (blur, lost car)? Deterministic encode (mean) for training data?
- Network: 3 stride-2 stages (64 → 32 → 16 → 8), channel widths, residual blocks, normalisation (GroupNorm), activation, decoder mirror; parameter budget (a few M at most) and speed from ticket 01.
- Loss: L1 vs L2 on pixels; **car-aware weighting** — the car is about 0.7% of the pixels. Options: weight pixels by a car mask (known exactly: render with and without the car), weight by difference from the static background, or nothing. Any adversarial term is out unless the prototype shows a clear need.
- Inputs/outputs: uint8 → float scaling, output activation, CHW conversion (model-side, per Step 3).
- Latent normalisation for Step 7: per-channel mean/std (or a single scale factor, as in latent diffusion) computed on train after freezing.
- Prototype: train two or three variants briefly on `frame_dataset(train)` and compare them with the ticket 04 metric.

Check: a chosen architecture (layer table, parameter count), loss with weights, latent normalisation rule, and prototype numbers on the ticket 04 metric that justify the choice.

## Comments
- Q1 plain AE vs VAE → **VAE with a small KL weight** (smooth, roughly Gaussian latents for Step 7; a decoder that tolerates slightly-off predicted latents in Step 8). Steps 6–8 use the **mean** latent (no sampling). The KL weight is tuned by the prototype; weight 0 (in effect a plain AE) is one of the variants, and wins only if the car survives nowhere else.
- Q2 loss → **L1** on pixels scaled to [−1, 1], with **car weighting**: weight `1 + λ` inside the car region (pixels where the Frame differs from the empty-track render in the episode's colours, grown by 1 pixel — same as ticket 04), `1` elsewhere. λ ≈ 30 makes the car region weigh about as much as the background. Training-only (privileged mask); encode/decode don't need it. No perceptual or adversarial terms.
- Q3 network → encoder: 3×3 conv → 32; [residual block, stride-2 conv] at 32 → 64 → 128 channels (64 → 32 → 16 → 8 px); residual block at 8×8×128; GroupNorm, SiLU, 1×1 conv → 32 channels (16 means + 16 log-variances). Decoder mirrors it with nearest-neighbour upsampling + 3×3 conv (no transposed convs: checkerboards hurt small details), ending GroupNorm, SiLU, 3×3 conv → 3. Residual block = (GroupNorm, SiLU, 3×3 conv) × 2 + skip. No attention. About 1.5 M params; "bigger" variant = 2 residual blocks per stage (about 3 M). Input `x / 127.5 − 1`; output clamped to [−1, 1] only when converting to uint8. Sizes subject to ticket 01 speed numbers.
- Q4 latent normalisation → **per-channel mean and std** (16 + 16 numbers) of the mean latents over a fixed sample of 65,536 train Frames, computed after freezing and stored in the checkpoint; `encode()` returns normalised latents, `decode()` un-normalises first. The prototype reports per-channel spreads.
- Q5 prototype → equal-budget runs scored with the full ticket 04 check, plus a **noisy-latent test** (decode normalised latents + 0.1·N(0, 1), check again):

  | Run | KL weight | λ | Network |
  |---|---|---|---|
  | A | 0 | 0 | small |
  | B | 0 | 30 | small |
  | C | 1e-4 | 30 | small |
  | D | 1e-3 | 30 | small |
  | (E) | best of C/D | 30 | bigger — only if none of A–D passes |

  KL weight is per latent value relative to the mean L1. **Selection:** must pass ticket 04 on clean latents → prefer passing on noisy latents → prefer the larger KL weight → prefer higher car-region PSNR.

- Prototype round 1 (2026-11-04, 6000 steps × 128 Frames ≈ 28–32 min each on MPS, 1.84 M params; outputs in `prototypes/out/vae_A…D/`). Clean latents, all 4096 val Frames:

  | Run | found | pos p95 / max (m) | head p95 / max (°) | car PSNR | vs true head p95 | clean / noisy |
  |---|---|---|---|---|---|---|
  | A | 14.5% | 1.79 / ∞ | 180 / 180 | 12.1 | — | FAIL / FAIL |
  | B | 100% | 0.21 / 0.39 | 12.2 / 35.3 | 35.7 | 13.9 | FAIL / FAIL |
  | C | 100% | 0.19 / 0.43 | 11.5 / 38.0 | 35.8 | 13.8 | FAIL / FAIL |
  | D | 100% | 0.18 / 0.36 | 10.9 / 31.7 | 35.7 | 12.8 | FAIL / FAIL |

  Readings: without car weighting (A) the car is erased; λ = 30 is essential. B–D pass position comfortably and miss only on heading, just above the limits (D: 10.9° vs 10°, 31.7° vs 30°). The `fast` slice passes for C and D; the `kerb` slice is worst (head p95 23–28°, 51 Frames). Noisy latents cost almost nothing. KL 1e-3 is not worse than 0 — D is best on heading. For reference, the estimator itself on originals vs the true State is heading p95 9.2° / max 31°, so the misses are at the estimator's own resolution; worst-heading cases on the contact sheet are a nose that is a fraction of a pixel off. Losses were still falling at the end. Probes (report only): colour Probes degrade heavily (brown_grass / light_road car PSNR 17–24 dB, heading p95 50–105°); the decoder repaints the training colours — expected for Step 10.
- Next per the selection rule: none of A–D passes → run **E** (bigger network) with KL **1e-3** (D's weight: best on every metric, and the larger KL is preferred).

**Status:** design decided (2026-11-03); round 1 done (2026-11-04), run E pending.
