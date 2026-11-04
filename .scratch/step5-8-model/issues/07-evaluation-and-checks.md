# Evaluation: what "looks like the expert's" means

Labels: wayfinder:grilling, wayfinder:prototype
Status: closed
Assignee: mgrue
Blocked by: 02-tokens-and-positions.md
Parent: ../MAP.md

## Question

How do we measure that training worked (Step 7) and that sampled chunks look like the expert's (Step 8), before closed-loop driving?

- Step 7: per-stream train and val losses falling; what counts as "falls" (e.g. val loss at end < x% of start, both streams)?
- Step 8 on fixed val Windows: action error of sampled chunks vs the clean labels (per step k, steer and throttle; mean over several samples vs best-of-n), excluding or separating the two-valid-answers Windows.
- The Fork with Instruction `any`: sample many chunks per Window; do they split into two clusters (outer / inner) rather than one average? Measured how (e.g. steer sign / predicted path through the Branch)? Same for `outer` / `inner`: do they follow the Instruction?
- Imagined future: decode predicted Latents and score them with `wam.carcheck` against the true future Frames (pose error by k).
- Baselines for comparison: zero-action, the expert's executed action noise level, the regression baseline (Step 10, can be cheap here).
- Thresholds: which numbers PASS/FAIL, which are report-only; a contact sheet / plot of sampled chunks vs labels.

Check: a list of metrics with thresholds (gate vs report), the val subset they run on, and validation of the metric on fakes (labels themselves pass, shuffled labels fail, averaged two-branch chunks fail the commit test).

## Comments
- From 02: `p_uncond = 0.1` by default, so every Policy supports CFG; `future_stride` is a training switch; no-video-loss mechanics are decided in 04.
- From 04: Fork data (train, `any`, approaching the fork section, steer × Direction sign): outer Chunk 0.52 → −0.78, inner 0.57 → 0.82, d′ 0.3 (k=1) → 1.6 (k=8) → 11 (k=16) — use the late-k steer / Chunk-mean steer as the commit statistic; short `horizon` shows less of the effect. Caveat: `video_stream = false` also has less compute per action. Regression is compared against flow + `video_stream = false` (same tokens).
- From 05: define (a) the fixed 256-Window val subset for periodic sampling metrics (incl. Fork / `any` Windows), (b) which metrics are gates for promoting `wam-v1` (PLAN Step 8), (c) the end-of-Run contact sheet. The Step 7 loss check is defined in 05 (both Streams ≤ 0.5× step-0 val and ≤ 1.05× min). Step 10 variants are recorded, not gated.
- From 06: sampling cost on MPS — 21.5 ms per Chunk batched (256 Windows × 10 steps ≈ 5.5 s), so `n_samples = 16` on 256 Windows ≈ 90 s; CFG adds ≈ 33 % at B = 1. Measure quality vs `n_steps ∈ {2, 4, 10, 20}` on the trained model. `Policy.act(..., n_samples, seed)` gives the Fork spread; `return_latents` + `compressor.decode` the imagined Frames.
- Prototypes (labels only, no model): `prototypes/fork_metric.py` (out `fork_metric.json`, `_stdout.txt`) and `prototypes/action_baselines.py` (out `action_baselines.json`, `_stdout.txt`).
- Fork reference Chunks: train label Chunks of Windows approaching the fork section, steer × Direction sign (+1 cw, −1 ccw), grouped by entry offset e = first k ∈ 1..16 with `branch[t+k] ≠ −1`, mean per (e, Planned branch) → 16 × 2 × 16 values. A Chunk is classified by the nearer reference (RMS over the 16 steer values) and committed if r = d_near / (d_near + d_far) < 0.3. Reference separation (RMS) 1.62 at e = 1 → 0.85 at e = 12 → 0.20 at e = 16; label accuracy 100 % for e ≤ 15, 91 % at e = 16 → use **e ≤ 12** (val: 5,514 Windows, 1,890 `any`). Fakes at e ≤ 12, R = 0.3: labels classified 100 %, committed 99.95 %; average of the two references committed 0 %; shuffled labels classified 50 %, committed 63 % (so commit alone is never the test).
- General baselines on 970 Windows (`eval_subset(val, 1024)` minus approaching-section), RMSE steer / throttle: train mean 0.61 / 0.39, zero 0.61 / 0.40, shuffled 0.86 / 0.54, labels + N(0, 0.1) 0.10 / 0.10; executed = clean on clean-future Windows (no Expert noise floor).
- Grilled 2026-11-04 (Q1–Q4). Step 7 loss check is 05's (both Streams ≤ 0.5× step-0 val, ≤ 1.05× min).
- Q1 Fork metrics (e ≤ 12, `n_samples = 16`): commit rate (fraction of samples with r < 0.3); following rate for `outer` / `inner` (fraction classified as instructed Branch); `any` balance (fraction of `any` samples classified outer); `any` two-sided (fraction of `any` Windows whose 16 samples contain both Branches). References computed once from train labels, frozen as eval data (not in the Checkpoint).
- Q2 general action error: 970-Window general set, one sample per Window (EMA, 10 steps, no CFG, fixed seed), RMSE vs clean label in raw units per dimension over all 16 steps and over k ≤ 4. Gate `rmse_steer ≤ 0.15`, `rmse_throttle ≤ 0.15`. Reported: best-of-16, per-k curves, baselines (train mean, zero, shuffled, regression variant).
- Q3 Fork subsets and gates: end-of-Run Fork set = 256 `any` + 128 `outer` + 128 `inner` val Windows, fixed seed, stratified over e = 1..12 (512 × 16 samples ≈ 3 min). Periodic set (every 8,000 steps, report-only) = 192 general (1 sample) + 64 Fork (32 / 16 / 16, 16 samples) ≈ 26 s. Gates: commit ≥ 0.90, following ≥ 0.90, `any` balance in [0.30, 0.70], `any` two-sided ≥ 0.80. Regression is expected to fail commit — recorded, not gated (05).
- Threshold discipline: gates set before any trained model; may be tightened once from the first 3,200-step quick Run before the full `wam-v1` Run, never loosened after a full-Run result; the card records the thresholds used.
- Q4 report-only: imagined future at end of Run on 256 general Windows — decode predicted Latents with vae-v1, `carcheck.score_frame(true_future, imagined, state_{t+k}, …)` → median / p95 position and heading error per k ∈ {2, …, 16}, next to floors (vae-v1 reconstruction of the true future Frame; the true Frame itself); omitted without video. Contact sheet `samples.png`: (1) 6 general Windows — obs Frame, true future k = 4, 8, 12, 16 and imagined below; (2) the same — steer and throttle vs k, label vs 16 samples; (3) 6 `any` Fork Windows — 16 Direction-normalised steer samples over the two references. `metrics.json` also gets best-of-16 and per-k RMSE, a CFG sweep w ∈ {1, 1.5, 2, 3} (following and commit), an `n_steps` sweep {2, 4, 10, 20} (general RMSE, commit). CLI `eval <run | checkpoint>` prints the gate table, exit 0 PASS / 1 FAIL, `--json`.

**Resolution (2026-11-04):**

| Metric | Set | Gate (`wam-v1`) | Validated on fakes |
|---|---|---|---|
| Step 7: val loss per Stream | 1024 val, deterministic (05) | ≤ 0.5× step-0 and ≤ 1.05× min | — |
| `rmse_steer`, `rmse_throttle` (1 sample) | 970 general val | ≤ 0.15 each | train mean 0.61 / 0.39, shuffled 0.86 / 0.54 fail; labels 0 pass |
| Commit rate | 512 Fork, e ≤ 12, × 16 | ≥ 0.90 | labels 0.9995 pass; averaged 0.00 fail |
| Following rate | 256 `outer` / `inner` × 16 | ≥ 0.90 | labels 1.00 pass; shuffled 0.50 fail |
| `any` balance | 256 `any` × 16 | 0.30–0.70 | — |
| `any` two-sided | 256 `any` × 16 | ≥ 0.80 | one-Branch collapse fails |
| Imagined-future pose error per k, best-of-16, per-k RMSE, CFG and `n_steps` sweeps | as above | report only | — |

All sampling with EMA weights, 10 Euler steps, no CFG, fixed seeds. Step 10 variants get the same report, never gated.

