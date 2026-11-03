# Architecture and training objective

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
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
