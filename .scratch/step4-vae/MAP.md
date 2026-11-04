# Step 4 frame compressor spec

Labels: wayfinder:map
Status: done (2026-11-04) — destination reached: SPEC.md
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step4-vae/SPEC.md`: a buildable spec for PLAN.md Step 4 (a small convolutional autoencoder that maps a 64×64 Frame to an 8×8 grid of 16-channel latents, trained on recorded Frames, then frozen) that an agent can implement without follow-up questions — **including the interface contract** Steps 6–10 rely on: the encode/decode API and latent layout, latent normalisation for flow matching, the frozen-checkpoint format and its provenance, and how Steps 6–8 get latents for Windows (cached or encoded on the fly).

PLAN.md check to satisfy: "reconstructions keep the car's position and heading clearly visible" — turned into a measurable check, not only a visual one.

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `CONVERSATION_SUMMARY.md`). Builds on the implemented Step 3 dataset (`.scratch/step3-dataset/SPEC.md`, `src/toycar/dataset.py`, recorded at `data/step3-v1`).
- Vocabulary follows repo-root `CONTEXT.md`; new terms (Latent, Checkpoint, …) are added there as tickets resolve.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` where a question needs measuring, research subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research findings under `research/`, throwaway prototypes under `prototypes/`) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no model code in `src/` beyond throwaway prototypes.
- Facts from Steps 1–3 that constrain the compressor:
  - Frames are 64×64×3 uint8 HWC, re-rendered from stored States (`frame_dataset(es)`, about 15k random Frames/s per process). 6000 episodes × 301 = 1.806M Frames; train 4800 × 301 = 1.445M. Consecutive Frames are nearly identical, so the effective variety is much smaller than the count.
  - Every Frame is the same fixed Track with per-episode colour jitter (5 colours, ×U[0.85, 1.15] brightness, ±10 per channel); the only moving thing is the car, 4.4 × 2.5 m ≈ 7 × 4 pixels (about 0.7% of the image), red body with a yellow nose marker. A plain pixel loss can score well while blurring the car away — PLAN and `CONVERSATION_SUMMARY.md` both flag "the car being tiny in latent space".
  - 8×8×16 = 1024 latent values per Frame vs 12,288 pixel values (12×). One latent cell covers 8×8 pixels = 5 × 5 m, so the car lives in one or two cells.
  - The exact State of every Frame is known (privileged), so car position/heading recovery from reconstructions can be measured against ground truth.
  - Probes (`brown_grass`, `light_road`, `blue_car`, `no_kerb`) re-render any Split with out-of-training colours; never train on them. Step 10 needs to know how the compressor behaves on them.
  - Step 6 token budget: 64 latent tokens per Frame, about 1100 tokens per Window; batch about 8 on this Mac with plain attention.
  - `toycar` stays numpy-only (Step 3 decision): model code and torch live outside it. The Step 3 `test_dataloader_compat` test runs once torch is installed.
  - Hardware: Apple M1 Pro, 16 GB RAM, about 50 GB free disk; Python 3.14 pinned (`.python-version`, `requires-python >=3.14,<3.15`); PyTorch on MPS.
  - Measured in Step 3 research: a 6.3 M-param transformer on MPS consumes about 13 Windows/s.

## Decisions so far

<!-- one line per closed ticket -->

- [PyTorch on Python 3.14 and MPS](issues/01-torch-on-mac.md) — torch 2.15.0 cp314 wheel from PyPI works; MPS ~11× CPU (~350–490 Frames/s at ~2 M params, ~1 h/epoch); float32 eager, batch 64–128, `num_workers=0`; seeded MPS runs bit-identical; encoder ≈ 5 ms per Window
- [Architecture and training objective](issues/03-architecture-and-objective.md) — VAE 64×64×3 → 8×8×16, channels 32/64/128/128, 2 residual blocks per lower stage, no attention, 3.17 M params; L1 with car region weighted 1 + 30, KL 1e-3; per-channel latent normalisation; prototype run E (12k steps, 89 min) the only one passing ticket 04 clean and noisy (pos p95 0.12 m, head p95 8.0° / max 24.7°)
- [Training recipe and freezing](issues/05-training-recipe.md) — run E's recipe unchanged: uniform random train Frames via `frame_dataset`, AdamW 1e-3, 200 warm-up + cosine to 0.05×, batch 128, 12,000 steps, float32, no EMA; ≈ 1 h 35 min; no early stopping, `val_rec` every 1,000 steps; resumable; freeze = latent stats → ticket 04 check → promote, never fine-tuned; a failed check reopens ticket 03 (no seed rerolls)
- [Latents for Steps 6–8](issues/06-latents-for-windows.md) — encode on the fly, no cache (6.1 ms per Window on MPS ≈ 8% of a Step 6 step); `load(name, device)` → frozen Compressor with `encode` (uint8 HWC, any leading dims → normalised float32 (…, 16, 8, 8) mean latents), `decode` (→ uint8, or differentiable float), `encode_windows`; row-major 64-token order; same path for Probes and closed loop; bit-identical per call, ≈ 1e-5 across batch sizes
- [Compressor API, CLI, tests and acceptance](issues/07-api-cli-tests.md) — `wam/compressor.py` (network, Compressor, load, promote, train, CLI `train` / `eval` / `show` / `promote`, exit 0 = clean PASS) and `wam/carcheck.py` (the ticket 04 check, reusable for Steps 8 / 10); 15 CPU tests with a tiny arch (< 60 s) + DataLoader + skip-if-absent real-checkpoint test; 6-step manual acceptance ending in a committed `vae-v1`
- [Compile SPEC.md](issues/08-compile-spec.md) — `SPEC.md` written; 12 clarifications, incl. one additive Step 3 method `FrameItems.locate(i)`
- [Measuring car visibility](issues/04-car-visibility-check.md) — colour-centroid pose estimator, reconstruction vs original on 4096 fixed val Frames: car found 100%, position p95 ≤ 0.3 m / max ≤ 0.6 m, heading p95 ≤ 10° / max ≤ 30°; PSNR (whole, car region, background) as diagnostics; Probes report-only; ~30-entry contact sheet; validated on six fake reconstructions (8×8 block average fails on heading); amended 2026-11-04: Frames whose original the estimator misreads by > 20° are left out (`skip`, > 1% → INVALID)
- [Package layout, runs and checkpoints](issues/02-package-layout-and-checkpoints.md) — second module `src/wam/` in the same uv project, torch in a default `model` group, test that `toycar` never imports torch; runs in `runs/vae/<date>-<name>/` (CSV/JSON/PNG, no TensorBoard); promoted, committed `checkpoints/vae-v1/{model.pt, card.json}` with provenance (commit, code hash, manifest hash, config, ticket 04 verdicts, repo-relative paths); load by name; bit-identical only on same device + torch

## Not yet specified

_(empty — every known question has a ticket)_

## Out of scope

- The Step 6 transformer, Step 7 training and Step 8 sampler — only the latent contract they consume is in scope.
- Temporal compression across Frames (FLUX's video VAE downsamples time 4×): PLAN fixes one latent grid per Frame, and action k ↔ future Frame k needs per-Frame latents. Revisit only if [ticket 03](issues/03-architecture-and-objective.md) finds a reason.
- Pretrained or perceptual losses (no downloaded networks).
- Changes to the Step 1–3 code, unless a ticket finds a blocking gap (then reopen the relevant ticket).
- Implementing the compressor (that's what the spec hands off).
