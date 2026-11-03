# Measuring "car position and heading clearly visible"

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

How do we turn the PLAN check into numbers, so architecture choices (ticket 03) and the frozen compressor can be judged objectively?

- Car pixels: the exact car mask of any Frame is available (render with the episode's colours, with and without the car). Candidate metrics: PSNR/MSE inside the car mask vs the whole Frame; whether the red body and yellow nose survive (colour-classified pixel counts).
- Pose recovery: estimate position and heading from a reconstruction and compare with the stored State. Options: colour centroids of body and nose pixels (heading = direction from body centroid to nose centroid); fitting the renderer to the reconstruction; a small probe network trained on reconstructions. Which is robust and simple enough, and does it also work on original Frames (as a baseline)?
- Thresholds: what error counts as "clearly visible" (e.g. position error well under a car width, heading error under the 180° ambiguity that a missing nose would cause), measured on which Frames (val, all ticks or the `eval_subset` Windows' Frames), and per-Probe reporting for Step 10.
- The visual half: a contact sheet of originals vs reconstructions (including fork-section and fast Frames, and each Probe).

Check: the metric definitions, the thresholds that pass PLAN's check, the evaluation set, and a baseline measurement of the pose estimator on original val Frames.
