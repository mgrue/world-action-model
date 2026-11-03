# Latents for Steps 6–8: cache or encode on the fly

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-torch-on-mac.md, 02-package-layout-and-checkpoints.md, 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

How do Steps 6–8 get normalised latents for each Window (17 Frames)?

- Encode on the fly on MPS inside the training loop (the encoder is small; Step 3 renders about 900 Windows/s vs about 13 Windows/s consumed) vs a latent cache next to the dataset (8×8×16 float16 = 2 KB per Frame, about 3.7 GB for all 6000 × 301 Frames, about 2.9 GB for train). Measure the encoder cost per Window from ticket 01 numbers.
- If cached: file layout (per Split, aligned with `states`), dtype, the mean vs a sampled latent (VAE), memory-mapping vs RAM, and how it is tied to the checkpoint and the dataset manifest so a stale cache is detected.
- Probes (Step 10) and Step 9 closed-loop frames are never cached — they need on-the-fly encoding anyway; does that argue for one path everywhere?
- The latent layout handed to Step 6: shape and order (e.g. (16, 8, 8) per Frame, normalised), and the encode/decode API the later steps call.

Check: the decision with measured per-Window cost, the API signatures, and (if cached) the file format and its staleness check.
