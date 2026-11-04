# Measuring "car position and heading clearly visible"

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

How do we turn the PLAN check into numbers, so architecture choices (ticket 03) and the frozen compressor can be judged objectively?

- Car pixels: the exact car mask of any Frame is available (render with the episode's colours, with and without the car). Candidate metrics: PSNR/MSE inside the car mask vs the whole Frame; whether the red body and yellow nose survive (colour-classified pixel counts).
- Pose recovery: estimate position and heading from a reconstruction and compare with the stored State. Options: colour centroids of body and nose pixels (heading = direction from body centroid to nose centroid); fitting the renderer to the reconstruction; a small probe network trained on reconstructions. Which is robust and simple enough, and does it also work on original Frames (as a baseline)?
- Thresholds: what error counts as "clearly visible" (e.g. position error well under a car width, heading error under the 180° ambiguity that a missing nose would cause), measured on which Frames (val, all ticks or the `eval_subset` Windows' Frames), and per-Probe reporting for Step 10.
- The visual half: a contact sheet of originals vs reconstructions (including fork-section and fast Frames, and each Probe).

Check: the metric definitions, the thresholds that pass PLAN's check, the evaluation set, and a baseline measurement of the pose estimator on original val Frames.

## Comments
- Q1 deciding measure → **pose recovery decides pass/fail**; pixel errors (car region, whole Frame, background) are reported as diagnostics.
- Q2 reference → compare the pose estimated on the **reconstruction with the pose estimated on the original Frame** (isolates what the compressor lost; estimator bias cancels); error vs the true State reported alongside.
- Q3 thresholds (reconstruction vs original) → car found in **100%** of Frames; position difference **p95 ≤ 0.3 m, max ≤ 0.6 m** (half a pixel typically, never a whole pixel); heading difference **p95 ≤ 10°, max ≤ 30°**; a missing nose counts as 180°. Judged on the tail, not the median.
- Q4 evaluation set → **4096 fixed val Frames**: episode i gets `4096 // E + (1 if i < 4096 % E else 0)` ticks drawn without replacement from 0 … length with `default_rng([seed_i, 5])`. Breakdown by situation: in the fork section (`branch != −1`), fast (v ≥ 8 m/s), off the Route (≥ 0.5 m from both centerlines; see clarification below). Test split untouched (Step 8).
- Q5 Probes → **report only**: the same 4096 Frames re-rendered per colour Probe, own table, never pass/fail. The estimator uses the Probe's colours.
- Q6 diagnostics → PSNR (dB, 0–255) over the whole Frame, the **car region** (pixels where the original differs from the empty-track render, grown by 1 pixel), and the background (the rest). No thresholds.
- Q7 contact sheet → one PNG, ~30 entries of *original | reconstruction | zoom of each around the car* with the estimated heading arrow and the measured differences: 8 random, 8 worst (4 by position, 4 by heading), 2 per situation, 2 per Probe.
- Q8 robustness → the estimator on a reconstruction searches only **±4 m around the original's estimate**, ignores pixels whose summed |ΔRGB| from the empty track is < 9, and needs a nose inside the square; ghost cars elsewhere show up in background PSNR and the contact sheet.
- Q9 validating the check → six hand-made fake reconstructions measured on the 4096 Frames (below); they become automated tests in ticket 07.

**Resolution (2026-11-03):**
- **Estimator** (`prototypes/pose_estimator.py`): car evidence = difference from the empty-track render in the episode's colours (weight `clip(Σ|ΔRGB| / 60, 0, 1)`, zero below 9); position = weighted centroid; nose = car pixels closer to the nose than the body colour; heading = direction from car centroid to nose centroid. On the 4096 original val Frames vs the true State: position median 0.05 / p95 0.16 / max 0.33 m; heading median 2.4° / p95 9.2° / max 31°; car and nose found in every Frame.
- **Pass/fail:** car found 100%; position difference p95 ≤ 0.3 m and max ≤ 0.6 m; heading difference p95 ≤ 10° and max ≤ 30° — reconstruction vs original, on the 4096 fixed val Frames. Diagnostics: PSNR whole / car region / background; error vs true State; per-situation and per-Probe tables; contact sheet.
- **The check catches what it should** (output in `prototypes/out/pose_check_baseline.txt`):

  | Fake reconstruction | found | pos p95 / max (m) | head p95 / max (°) | PSNR all / car / bg | Verdict |
  |---|---|---|---|---|---|
  | exact copy | 100% | 0 / 0 | 0 / 0 | ∞ | PASS |
  | car shifted 1 px | 100% | 0.625 / 0.641 | 0 / 14 | 33.9 / 16.3 / ∞ | FAIL (position) |
  | car erased | 0% | – | 180 / 180 | 29.6 / 12.0 / ∞ | FAIL (found) |
  | car facing backwards | 100% | 0.12 / 0.20 | 180 / 180 | 31.1 / 13.6 / ∞ | FAIL (heading) |
  | background noise ±4/channel | 100% | 0.13 / 0.23 | 7 / 20 | 39.9 / 39.9 / 39.9 | PASS |
  | 8×8 block average, enlarged | 100% | 0.60 / 0.93 | 180 / 180 | 22.5 / 13.8 / 23.1 | FAIL (both) |

  The naive 8×8 "compressor" fails clearly on heading (the nose is averaged away) — the failure PLAN worries about.
- **Clarification made while running:** the off-Route slice uses ≥ 0.5 m, not 1 m — only 2 of 4096 expert Frames are ≥ 1 m off (≥ 0.5 m: 51, ≥ 0.3 m: 122). Recovery coverage is thin, as Step 3 §13 noted.
- **Side note for ticket 03:** heading is the fragile part (the nose is ~2 × 4 pixels); background noise of ±4 per channel alone already moves the heading max to 20°, so the car-aware loss weighting deserves a real look.

**Amendment (2026-11-04, from the ticket 03 prototype, agreed with the user):** a Frame counts only if the estimator can read its **original**: Frames where the estimate on the original is more than **20°** from the true heading are left out of every statistic and reported as `skip`; if more than **1%** of the Frames are left out, the check is **INVALID** (not PASS). Reason: ticket 03's run E (12,000 steps) failed heading max on a single Frame (val episode 223, tick 7) whose reconstruction is 0.0° from the true heading — the 31.4° came from the estimator misreading the original (the 31.4° baseline max above). The limits themselves are unchanged. Of the 4096 Frames, exactly that one is left out (2 would be at 15°, 1 at 25°). The six fakes keep their verdicts (re-run in `prototypes/out/pose_check_baseline.txt`); `summarise` / `readable` in `prototypes/pose_estimator.py` implement it.
