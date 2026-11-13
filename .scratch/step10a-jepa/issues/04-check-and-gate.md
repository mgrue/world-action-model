# Check and gate

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-linear-probe-on-vae-v1.md
Parent: ../MAP.md

## Question

When does `jepa-v1` PASS, and what is only reported?

- Probe thresholds: absolute (pos p95 ≤ 0.3 m, heading p95 ≤ 10°, the Step 4 bar) or relative to what the same probe reads from `vae-v1` (ticket 01)? Max limits too, as in Step 4?
- Collapse gate: which diagnostic(s) and threshold (e.g. every channel's std across Frames above some floor, effective rank), logged during training and checked at the end.
- Report only: colour Probes, situation slices (`section`, `fast`, `kerb`), the probe on `vae-v1` side by side.
- Evaluation Frames and the probe's train/eval split (fixed seeds, same Frames for both encoders).
- PASS / FAIL / INVALID and CLI exit codes, matching `wam.compressor eval`.

Check: a written gate with numbers, the evaluation set, and the report table layout; the ticket-03 prototype result scored against it if available.

## Comments
