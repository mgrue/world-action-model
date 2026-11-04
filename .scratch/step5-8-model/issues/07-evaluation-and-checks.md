# Evaluation: what "looks like the expert's" means

Labels: wayfinder:grilling, wayfinder:prototype
Status: open
Assignee:
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
