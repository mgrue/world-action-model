# Latents for Steps 6–8: cache or encode on the fly

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 01-torch-on-mac.md, 02-package-layout-and-checkpoints.md, 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

How do Steps 6–8 get normalised latents for each Window (17 Frames)?

- Encode on the fly on MPS inside the training loop (the encoder is small; Step 3 renders about 900 Windows/s vs about 13 Windows/s consumed) vs a latent cache next to the dataset (8×8×16 float16 = 2 KB per Frame, about 3.7 GB for all 6000 × 301 Frames, about 2.9 GB for train). Measure the encoder cost per Window from ticket 01 numbers.
- If cached: file layout (per Split, aligned with `states`), dtype, the mean vs a sampled latent (VAE), memory-mapping vs RAM, and how it is tied to the checkpoint and the dataset manifest so a stale cache is detected.
- Probes (Step 10) and Step 9 closed-loop frames are never cached — they need on-the-fly encoding anyway; does that argue for one path everywhere?
- The latent layout handed to Step 6: shape and order (e.g. (16, 8, 8) per Frame, normalised), and the encode/decode API the later steps call.

Check: the decision with measured per-Window cost, the API signatures, and (if cached) the file format and its staleness check.

## Comments
- Measured (2026-11-04, run E's encoder, MPS, float32, batch of 8 Windows = 136 Frames): Step 3 load + render 18 ms, encode 48 ms → **2.3 + 6.1 ms per Window**; against ≈ 615 ms per batch-8 step of the ticket 01 stand-in transformer, on-the-fly encoding adds ≈ 8% (≈ 11% with rendering). A float16 cache would be 3.7 GB (2.9 GB train) and ≈ 11 min to build. Script: scratchpad benchmark loading `prototypes/out/vae_E/model.pt` over random train Windows.
- Q1 cache vs on the fly → **encode on the fly, no cache.** One code path for training (Steps 6–7), Probes (Step 10) and closed loop (Step 9); nothing on disk to go stale; latents always float32. A ~10% speed-up does not pay for a format, a staleness check and a second path, and one path guarantees training and closed loop see identical latents. A cache can be added behind the same API later if Step 7 turns out to be data-bound.
- Q2 API and layout → `wam.compressor.load(name, device) -> Compressor` (frozen: eval, `requires_grad=False`, latent stats loaded). `encode(frames)`: uint8 `(..., 64, 64, 3)` (torch or numpy, any leading dims) → float32 `(..., 16, 8, 8)` **normalised mean latents** on the compressor's device, under `inference_mode`. `decode(z)`: `(..., 16, 8, 8)` → uint8 `(..., 64, 64, 3)` under `inference_mode`; `decode(z, as_float=True)` → float32 `(..., 3, 64, 64)` in [−1, 1] **without** `inference_mode`, so gradients can flow to `z` (weights stay frozen). `encode_windows(batch)` → `(obs (B, 16, 8, 8), future (B, H, 16, 8, 8))` in one encoder call; future latent k ↔ action k. Attributes `name`, `card`, `latent_shape == (16, 8, 8)`, `device`. Token order for Step 6: `z.flatten(-2).transpose(-1, -2)` → 64 tokens × 16, **row-major**; token i = row `i // 8`, col `i % 8`, covering pixels `[8·row, 8·row + 8) × [8·col, 8·col + 8)`.
- Q3 Probes / closed loop / exactness → Probes via `frame_dataset(es, probe=…)` / `windows(probe=…)` and closed-loop Frames from the env's `render` use the same `encode`. Encoding runs in the main process on MPS after a batch is assembled (`num_workers=0`). Same call + same input → bit-identical; across batch sizes only within float tolerance (≈ 1e-5 normalised; CPU test: one Frame alone vs inside a batch agree within 1e-5) — nobody relies on bit-equality across batch sizes. Always the mean latent, never sampled after training.

**Resolution (2026-11-04):**
```python
from wam.compressor import load
vae = load("vae-v1", device="mps")          # frozen Compressor
z = vae.encode(frames)                      # uint8 (..., 64, 64, 3) → float32 (..., 16, 8, 8), normalised mean latents
x = vae.decode(z)                           # → uint8 (..., 64, 64, 3);  as_float=True → float32 (..., 3, 64, 64) in [−1, 1], differentiable in z
obs, fut = vae.encode_windows(batch)        # Step 3 Window batch → (B, 16, 8, 8), (B, H, 16, 8, 8)
tokens = z.flatten(-2).transpose(-1, -2)    # (..., 64, 16), row-major cells
```
No latent cache: on the fly everywhere (≈ 8% of a Step 6 step on MPS; 6.1 ms per Window).

