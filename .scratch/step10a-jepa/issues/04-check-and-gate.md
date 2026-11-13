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
- From 01: the gate uses the **MLP readout** (definition in [01](01-linear-probe-on-vae-v1.md)); `vae-v1` reference: pos p95 0.24 / max 2.7 m, head p95 1.7 / max 15° — passes the Step 4 p95 bar, fails its max limits. Decide absolute vs relative, whether max limits apply (or p99 instead), and measure the MLP readout's seed spread (fit seeds 0–2) before fixing thresholds. Ridge is report-only.
