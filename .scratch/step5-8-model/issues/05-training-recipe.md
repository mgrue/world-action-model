# Training recipe

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 01-throughput-budget.md, 03-transformer-blocks.md, 04-flow-matching-objective.md
Parent: ../MAP.md

## Question

How is the model trained, for how long, and when is a run good enough to freeze?

- Data: uniform over train Windows (797k), or oversample `approaches_section` / `any` Windows? Probes never. Colour jitter is already in the data.
- Instruction dropout for classifier-free guidance (PLAN Step 10: ~10%) — in every run, or a switch?
- Optimiser, lr, warm-up, schedule, batch, steps (within ticket 01's hours), weight decay, gradient clipping, EMA (FLUX keeps EMA copies)?
- Monitoring: train loss per stream (PLAN Step 7 check), val loss per stream on a fixed `eval_subset`, periodic sampling metrics (ticket 07) — how often?
- Seeds and determinism, resume (reuse the Step 4 pattern), Run directory layout `runs/wam/…`.
- Stopping / freezing rule and what a failed run triggers (Step 4: reopen design, no seed rerolls).

Check: a recipe table (data, optimiser, schedule, batch, steps, expected hours), monitoring list, and a freeze rule.

## Comments
- From 04: loss and switches fixed (see 04 Resolution). If `loss_action` stalls, sweep `action_loss_weight ∈ {1, 5}` in a 100k-Window quick run. Logs: `loss_video`, `loss_action`, `loss_action_hi_t` (t > 0.7), `loss_action_lo_t`. Action normalisation constants (steer /0.618; throttle −0.077, /0.373) go into the config.
- From 03: model 8.23 M params at d256×8; 236 ms/step at batch 32 on MPS (≈ 85 W/s end to end, ≈ 3.3 h per 1 M Windows). Overfit sanity run used AdamW 3e-4, grad clip 1.0, no warm-up, no weight decay — stable; a starting point, not a decision. Escalate to d384×8 only if the quick run underfits.
- Grilled 2026-11-04 (Q1–Q6); precedent `.scratch/step4-vae/SPEC.md` §5–6 (AdamW + warm-up + cosine, full schedule, no seed rerolls, resumable Runs). Instruction dropout already settled by 02 (`p_uncond = 0.1` in every Run).
- Q1 data: uniform over train Windows, no oversampling, no switch — ≈ 25k `any` Fork Windows per 1 M seen is enough; matches the closed-loop state distribution; regression and flow see identical data. Seeded numpy permutation, new each pass. Probes never; test untouched. Oversampling is the first reopen if 07's Fork metric fails.
- Q2 optimiser: batch 32; AdamW lr 3e-4, betas (0.9, 0.99), weight decay 0.01 on Linear weights only (not norms, Instruction embedding, biases); 1,000 linear warm-up steps then cosine to 0.05× at the last step (Step 4 formula; other `--steps` keep the warm-up and stretch the cosine); grad clip 1.0; fp32 eager SDPA, no compile. Full Run 32,000 steps (1.02 M Windows ≈ 1.3 passes, ≈ 3.4 h), quick Run 3,200 steps (≈ 20 min). No early stopping.
- Q3 EMA: decay 0.999 with warm-up `min(0.999, (1+s)/(10+s))`; val logged for raw and EMA; sampling metrics and the Checkpoint use EMA only.
- Q4 monitoring: every 100 steps `loss, loss_video, loss_action, loss_action_hi_t, loss_action_lo_t, grad_norm, lr, windows_per_s, elapsed_s`; every 2,000 steps (and at step 0) val per stream (+ hi/lo t) for raw and EMA on fixed `eval_subset(val, 1024)` with per-Window fixed t and noise (seeded), `p_uncond = 0`; every 8,000 steps and at the end 07's sampling metrics (EMA, 10 Euler steps) on a fixed 256-Window val subset (defined by 07); end: 07's full check and contact sheet. Variants leave video columns empty.
- Q5 Runs: Step 4 pattern — `runs/wam/<YYYYMMDD-HHMM>-<name>/` with `config.json` (all switches, action-normalisation constants, provenance: git sha, vae-v1 card hash, dataset manifest hash, resumes), `log.csv`, `metrics.json`, `stdout.txt`, PNGs, `checkpoint.pt`; one `seed` → numpy permutation, torch generator (t, noise, p_uncond), init; bit-identical on same seed/device/torch/code; `resume.pt` every 1,000 steps (model, EMA, optimiser, scheduler, step, sampler and generator state), deleted at the end, `train --resume`, resumed Runs promotable; test 20 steps == 10 + resume + 10 (CPU tiny). Additions: `checkpoint.pt` = EMA weights + arch + switches + normalisation constants (self-sufficient for a Policy; raw weights only in `resume.pt`); input pipeline in the main process (render → encode → step; DataLoader workers would save ≈ 10 % deterministically — documented, not built); full Runs under `caffeinate -i`.
- Q6 freeze: Step 7 check = for both Streams, final EMA val loss ≤ 0.5 × its step-0 value (≈ 2.0 by zero-init) and ≤ 1.05 × the Run's minimum val loss (gate). Promote `wam-v1` (default switches) when the schedule finished without NaN, the Step 7 check passes and 07's gate metrics pass → `checkpoints/wam-v1/` + `card.json`, committed, load by name. Step 10 variants (`wam-v1-regression`, `-novideo`, `-h8`, …) are experiments: promotable with their check results in the card even if failing. Every full Run is preceded by a 3,200-step quick Run that passes the Step 7 check. Failed `wam-v1` check: no seed reroll (only crash / NaN / interruption reruns); reopen by symptom: action loss stalls → `action_loss_weight 5`; underfit → d384×8; Fork commit fails → oversample Fork Windows; still improving → 2× steps; each a new quick then full Run.

**Resolution (2026-11-04):**

| | Full Run | Quick Run |
|---|---|---|
| Data | uniform over 797,262 train Windows, seeded permutation per pass | same |
| Batch / steps | 32 / 32,000 (1.02 M Windows) | 32 / 3,200 |
| Optimiser | AdamW 3e-4, betas (0.9, 0.99), wd 0.01 on Linear weights, clip 1.0 | same |
| Schedule | 1,000 warm-up, cosine to 0.05× | same |
| EMA | 0.999 (warm-up), Checkpoint + samples on EMA | same |
| Precision | fp32 eager, SDPA | same |
| Expected time | ≈ 3.4 h (`caffeinate -i`) | ≈ 20 min |

Monitoring: train every 100 steps, val (raw + EMA, deterministic, 1024 val Windows) at 0 and every 2,000, sampling metrics every 8,000 and at the end. Freeze: Step 7 check (both Streams ≤ 0.5× step-0 and ≤ 1.05× min) + 07 gates → `wam-v1`; no seed rerolls; symptom-ordered reopen list above.

