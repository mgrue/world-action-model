# Step 4 VAE: architecture and purpose

2026-11-04 · explainer for `.scratch/step4-vae/SPEC.md`

## Why the VAE exists

The world action model never works on pixels. It reads and predicts **Latents**: each 64×64×3 Frame (12,288 values) is compressed to an 8×8×16 grid (1,024 values), 12× smaller. The VAE is the frozen compressor that does this, trained once in Step 4 and never changed afterwards.

- **Smaller target.** Step 7 predicts the H = 16 future Frames of a Window with flow matching. Predicting 16 × 1,024 latent values is far cheaper than 16 × 12,288 pixels.
- **Tokens for the transformer.** The 8×8 grid becomes 64 tokens of 16 channels per Frame, each covering a 5 × 5 m patch of Track.
- **One fixed interface.** Steps 6–10 call `load("vae-v1")` and use the same `encode` / `decode`. A better compressor gets a new name (`vae-v2`) and everything downstream reruns.
- **Only the car matters.** The background is the same Track every time; the information the model needs is the car's position and heading. The compressor is judged on exactly that.

## Architecture

A small convolutional VAE (3,166,435 parameters, `src/wam/compressor.py`). The encoder halves the resolution three times (64 → 32 → 16 → 8); the decoder mirrors it.

```
        Encoder (downsample ×2 three times)          Decoder (upsample ×2 three times)

  Frame 64×64×3 → conv to 32 ch                 conv → reconstructed Frame 64×64×3
            │                                                  ▲
  32×32×64  · 2 res blocks                      64×64×32
            │                                                  ▲
  16×16×128 · 2 res blocks                      32×32×64  · 2 res blocks
            │                                                  ▲
  8×8×128   · 2 res blocks                      16×16×128 · 2 res blocks
            │                                                  ▲
  1×1 conv → μ and log σ², 8×8×16 each          conv 16 → 128 · 8×8 · 2 res blocks
            │                                                  ▲
            └──────►  Latent: 8×8×16 = 64 tokens of 16 channels ─┘
                      training: z = μ + σ·ε
                      after training: μ only, normalised per channel
```

- Res block = (GroupNorm, SiLU, 3×3 conv) × 2 plus a skip connection.
- Downsampling = stride-2 conv; upsampling = nearest-neighbour ×2 then 3×3 conv. No attention, no transposed convs.
- Channels 32 / 64 / 128 / 128, float32 on MPS.

The 16 latent channels and the 8×8 grid are fixed by the contract with later Steps; only channel widths and block counts could change in a future `vae-v2`.

## Training objective

The loss is a car-weighted pixel L1 plus a very small KL term, so the network spends its capacity on the car.

```
loss = Σ w·|x̂ − x| / (3 · Σ w)  +  1e-3 · KL,     w = 31 in the car region, 1 elsewhere
```

- **Car weighting (λ = 30).** The car is about 0.7% of the pixels. The car region (pixels that differ from the empty-track render, grown by 1 px) gets weight 1 + λ = 31, so it counts about as much as the whole background. The mask is used only in training; `encode` / `decode` never need it.
- **Small KL (10⁻³).** Keeps the latent space smooth, so the decoder tolerates the imperfect Latents that later Steps predict. The check with 0.1·N(0, 1) noise added to the Latents passes.
- **Latent normalisation.** After training, each of the 16 channels gets a mean and std over 65,536 train Frames. `encode` returns (μ − mean) / std, so Step 7's flow matching sees unit-variance Latents with no extra scale factor.
- **Recipe.** AdamW, lr 10⁻³, 200 warm-up steps then cosine down to 5%; 12,000 steps × 128 Frames (about 1.06 passes over the 1.445 M train Frames); about 1 h 35 min on an M1 Pro, including the final check.

## Why this design

Five prototype runs on the real dataset picked the design; only run E passed the car-visibility check, and its recipe is now the spec. Gate: car found in 100% of 4,096 val Frames, position p95 ≤ 0.3 m / max ≤ 0.6 m, heading p95 ≤ 10° / max ≤ 30°.

| Run | Change tested | Params | Steps | Position p95 / max (m) | Heading p95 / max (°) | Result |
| --- | --- | --- | --- | --- | --- | --- |
| A | plain L1, no KL | 1.84 M | 6,000 | car found in 14.5% | 180 / 180 | FAIL: car erased |
| B | + car weight λ = 30 | 1.84 M | 6,000 | 0.21 / 0.39 | 12.2 / 28.6 | FAIL |
| C | + KL 10⁻⁴ | 1.84 M | 6,000 | 0.19 / 0.43 | 11.5 / 38.0 | FAIL |
| D | KL 10⁻³ | 1.84 M | 6,000 | 0.18 / 0.36 | 10.9 / 31.7 | FAIL |
| **E** | 2 res blocks per stage, KL 10⁻³ | **3.17 M** | **12,000** | **0.12 / 0.26** | **8.0 / 24.7** | **PASS** |

- **A → B:** without car weighting the network simply erases the car.
- **B → D:** position is fine; heading misses by about a degree. KL 10⁻³ was slightly best and costs nothing.
- **D → E:** a bigger network trained twice as long sharpens the yellow nose enough to pass, clean and with noisy Latents.
- Scores use the amended check, which drops val Frames whose *original* the estimator misreads by more than 20° (1 of 4,096).

Still weak, recorded but not blocking: cars off the Route (heading p95 about 17°) and the colour Probes. The decoder repaints training colours, so Probe failures in Step 10 may start here.

## How later Steps use it

Once `vae-v1` passes and is promoted to `checkpoints/vae-v1/`, every later Step goes through the same four calls.

| Call | Input → output | Used by |
| --- | --- | --- |
| `load("vae-v1")` | name → frozen `Compressor` (weights never trained again) | Steps 6–10 |
| `encode_windows(batch)` | Window Frames → obs (B, 16, 8, 8), future (B, 16, 16, 8, 8) | Steps 6–7 training |
| `encode(frames)` | uint8 (…, 64, 64, 3) → normalised μ (…, 16, 8, 8) | Step 9 closed loop, Step 10 Probes |
| `decode(z)` | Latents → uint8 Frames, or differentiable floats with `as_float=True` | Step 8, scored with `wam.carcheck.check` |

- **Encoded on the fly.** No latent cache: about 6 ms per Window (17 Frames) on MPS, under 11% of a training step.
- **Token order.** `z.flatten(-2).transpose(-1, -2)` gives (…, 64, 16), row-major: token i is grid cell (i // 8, i % 8).
- **Same check everywhere.** `wam/carcheck.py` knows nothing about the VAE, so Step 8 can score predicted Frames and Step 10 can score Probe Frames with the same thresholds.

Source: `.scratch/step4-vae/SPEC.md` and ticket 03 in the same folder.
