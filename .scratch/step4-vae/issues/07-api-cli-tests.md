# Compressor API, CLI, tests and acceptance

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 02-package-layout-and-checkpoints.md, 04-car-visibility-check.md, 05-training-recipe.md, 06-latents-for-windows.md
Parent: ../MAP.md

## Question

What module, API and commands deliver the compressor, and what proves it correct?

- API: model class, `load(name)` of the frozen checkpoint, `encode(frames_uint8) -> latents` (normalised), `decode(latents) -> frames_uint8`, batch/device handling.
- CLI: `train`, `eval` (ticket 04 metrics on val and each Probe, PASS/FAIL against the thresholds), `show` (contact sheet PNG), and `cache` if ticket 06 caches.
- Automated tests (fast, CPU, no full training): shapes and dtypes, encode/decode round trip on a tiny untrained model, checkpoint save/load, normalisation applied/undone, the pose estimator on original Frames, `toycar` still imports without torch.
- Manual acceptance: train the frozen model, `eval` passes, contact sheet shows the car's position and heading in reconstructions (PLAN check).

Check: an API and CLI sketch with signatures, the list of automated tests, and the manual acceptance steps.
