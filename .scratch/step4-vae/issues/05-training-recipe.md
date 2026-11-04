# Training recipe and freezing

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 03-architecture-and-objective.md
Parent: ../MAP.md

## Question

How is the compressor trained to the frozen version, within a weekend budget on this Mac?

- Data: all train Frames, or a subsample (consecutive Frames are near-duplicates)? Random-Frame sampling via `frame_dataset(train)`; val Frames for monitoring; Probes for reporting only.
- Optimiser, learning rate and schedule, batch size, number of steps / wall-clock budget, EMA weights or not, mixed precision on MPS (from ticket 01).
- Stopping rule and what "frozen" means: the checkpoint that passes the ticket 04 thresholds on val; the latent normalisation statistics computed afterwards on train.
- Reproducibility: seed, determinism expectations from ticket 01.
- What gets logged per run (losses, ticket 04 metrics on val, reconstruction images) — consistent with ticket 02.

Check: a full recipe table, an expected wall-clock time, and the freeze procedure.

## Comments
- Q1 data → each training Frame is drawn **uniformly over all train Frames** (`frame_dataset(train)`, 1.445 M items, with replacement), rendered on the fly; its car mask comes from the empty-track render in the episode's colours. No subsampling (random draws see each Frame about once anyway) and no oversampling of off-Route Frames (≈ 1% of data, not a gate; a dataset question for Steps 8–10). Val only for monitoring and the final check; Probes report-only.
- Q2 optimisation → **run E's recipe unchanged**: AdamW lr 1e-3, weight decay 1e-4; 200 linear warm-up steps then cosine to 0.05× (5e-5) at the last step (warm-up fixed, cosine stretched if `--steps` differs); batch 128; **12,000 steps** (1.54 M Frames ≈ 1.06 × train); float32 eager, no `torch.compile`, no mixed precision; **no EMA** (untested — the first thing to try for a future `vae-v2`). Expected ≈ 89 min training + ≈ 5 min latent stats and check ≈ **1 h 35 min** on the M1 Pro (≈ 290 Frames/s).
- Q3 stopping / freezing → **no early stopping**, always the full step count. Every 1,000 steps `val_rec` = reconstruction loss on 1,024 fixed val Frames. After the last step: (1) latent normalisation stats = per-channel mean / std of the mean latents over 65,536 train Frames from a fixed seed stream; (2) the full ticket 04 check on the 4096 val Frames (clean, noisy, per Probe); (3) contact sheet; (4) run directory written. **Frozen = promoted** (ticket 02; refused if the clean check fails): weights and stats never change afterwards; Steps 6–10 never fine-tune or jointly train the compressor; a better one is a new name and everything downstream reruns. A failed final check is **not** retried with other seeds — the recipe is not robust, reopen ticket 03 (EMA or extra nose weight first); only runs broken by a crash / NaN / interruption are rerun.
- Q4 interruption → **resumable**: every 1,000 steps `resume.pt` in the run directory (model, optimiser, scheduler, step, numpy sampler state, torch generator state for the VAE noise); `python -m wam.compressor train --resume runs/vae/<run>` continues (loses ≤ ~7 min); `resume.pt` is deleted when the run finishes. Same device + torch → aimed bit-identical to an uninterrupted run, tested on CPU (20 steps straight == 10 + resume + 10); `config.json` records resumes and their steps; resumed runs are promotable. Start long runs with `caffeinate -i uv run …` so the Mac does not sleep.
- Q5 logging → `log.csv` every 100 steps (step, loss, rec, KL, lr, Frames/s, elapsed) plus `val_rec` and the resume save every 1,000; terminal line with ETA, copied to `stdout.txt`; `metrics.json` with the full ticket 04 check (incl. `skip`), per-channel latent mean / std, training time, params, Frames/s; `contact.png` once at the end; no images during training. Near-unused latent channels (run E: std 0.11 / 0.08 on channels 6 / 15) are fine — normalisation gives Step 7 unit variance, and decoder-side noise shrinks for them — so only reported, no rule.

**Resolution (2026-11-04):**

| | Recipe |
|---|---|
| Data | uniform random train Frames via `frame_dataset(train)`, rendered on the fly, with car masks |
| Network / loss | ticket 03: VAE `blocks=2`, 3.17 M params; L1 with car region × (1 + 30), KL 1e-3 |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | 200 warm-up steps, cosine to 0.05× |
| Batch / steps | 128 / 12,000 (≈ 1.06 epochs of train Frames) |
| Precision | float32 eager on MPS; no compile, no EMA |
| Seed | one seed, separate derived streams (ticket 02, Q6) |
| Monitoring | `val_rec` on 1,024 fixed val Frames every 1,000 steps |
| Resume | `resume.pt` every 1,000 steps, `--resume` |
| Wall clock | ≈ 1 h 35 min (89 min training + stats + check) |

**Freeze procedure:** train the full schedule → latent stats on 65,536 train Frames → ticket 04 check (clean must PASS) → contact sheet → `promote … --name vae-v1` → committed under `checkpoints/` and never changed.

