# Architecture and training objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
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

- Run E (2026-11-04, KL 1e-3, λ 30, 2 residual blocks per stage, 3.17 M params, 41 min): clean — found 100%, pos p95 / max 0.15 / 0.35 m, head p95 / max **9.7 / 35.3°**, car PSNR 36.4; noisy — head p95 / max 10.1 / 36.2°. Fails on heading max only, because of **a single Frame** (val episode 240, tick 0, off the Route): it is the only one of 4096 over 25° in both clean and noisy, and the next worst are 24°, 23° and 22°. p99 is 15°. Against the true State, that Frame's reconstruction is 41.5° off. `section` and `fast` pass.

- Run E, 12,000 steps (89 min, same seed): clean — found 100%, pos p95 / max 0.12 / 0.26 m, head p95 / max 8.0 / 31.4°, car PSNR 38.3. The one Frame over 30° (val episode 223, tick 7) is reconstructed 0.0° from the true heading; the estimator misreads the original. This led to the **ticket 04 amendment** (Frames whose original the estimator misreads by > 20° are left out; exactly 1 of 4096). All runs re-scored under it (`--eval-only`, same model weights):

  | Run | skip | pos p95 / max (m) | head p95 / max (°) | car PSNR | clean / noisy |
  |---|---|---|---|---|---|
  | A | 1 | 1.79 / ∞ (found 14.5%) | 180 / 180 | 12.1 | FAIL / FAIL |
  | B | 1 | 0.21 / 0.39 | 12.2 / 28.6 | 35.7 | FAIL / FAIL |
  | C | 1 | 0.19 / 0.43 | 11.5 / 38.0 | 35.8 | FAIL / FAIL |
  | D | 1 | 0.18 / 0.36 | 10.9 / 31.7 | 35.7 | FAIL / FAIL |
  | **E, 12k steps** | 1 | **0.12 / 0.26** | **8.0 / 24.7** | **38.3** | **PASS / PASS** (noisy: 0.12 / 0.31 m, 8.2 / 25.4°) |

**Resolution (2026-11-04):** run **E** is the only variant that passes, clean and noisy.
- **Network** (`prototypes/vae_proto.py`, `VAE(blocks=2)`), 3,166,435 params; Res = (GroupNorm, SiLU, 3×3 conv) × 2 + skip; GroupNorm groups = min(32, c/4); no attention; nearest-neighbour upsampling, no transposed convs:

  | Encoder | out | Decoder | out |
  |---|---|---|---|
  | 3×3 conv 3 → 32 (no Res at 64×64) | 64×64×32 | 3×3 conv 16 → 128, 2 × Res | 8×8×128 |
  | 3×3 conv stride 2 → 64, 2 × Res | 32×32×64 | up ×2, 3×3 conv → 128, 2 × Res | 16×16×128 |
  | 3×3 conv stride 2 → 128, 2 × Res | 16×16×128 | up ×2, 3×3 conv → 64, 2 × Res | 32×32×64 |
  | 3×3 conv stride 2 → 128, 2 × Res | 8×8×128 | up ×2, 3×3 conv → 32 | 64×64×32 |
  | GroupNorm, SiLU, 1×1 conv → 32 (16 μ + 16 log σ², clamped −30 … 20) | 8×8×32 | GroupNorm, SiLU, 3×3 conv → 3 | 64×64×3 |

  This refines Q3: the stride-2 conv comes before the stage's residual blocks, and the 64×64 stage has no residual block (ticket 01 speed).
- **Loss:** L1 on pixels in [−1, 1], weight 1 + **λ = 30** in the car region (car mask grown by 1 px), weighted mean; plus **KL weight 1e-3** per latent value; z sampled in training only. Without car weighting (A) the car is erased.
- **Latent normalisation:** per-channel mean / std of the mean latents over 65,536 train Frames, stored with the weights (Q4). E's spreads range 0.08–1.01: channels 6 and 15 are nearly unused (std 0.11, 0.08), so normalisation scales them ~10×; the noisy-latent test passed regardless — ticket 05/06 to keep an eye on it.
- **Prototype recipe that produced it** (input to ticket 05): AdamW lr 1e-3, weight decay 1e-4, 200 warm-up steps, cosine to 0.05×; batch 128 random train Frames; 12,000 steps (1.54 M Frames ≈ 1.06 × the train Frames); 89 min on MPS at ≈ 290 Frames/s; seed 0. At 6,000 steps E missed (head p95 9.7°, max 35.3° on a readable Frame), so the step count matters.
- **Known weak spot:** off the Route (`kerb`, 51 Frames) head p95 17° — thin recovery data (Step 3 §13); report it in Steps 8 / 10, not a gate. Colour Probes degrade heavily (the decoder repaints training colours), as expected for Step 10.
- Outputs: `prototypes/out/vae_A … vae_E/` (metrics.json, contact.png, log.csv; model.pt gitignored).
