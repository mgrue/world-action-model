# Package layout, API, CLI, outputs and tests

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-replanning-scheme.md, 05-metrics-and-gates.md, 06-robustness-report.md
Parent: ../MAP.md

## Question

How is the driver packaged?

- Module(s) (e.g. `wam.drive`), the driver protocol shared by Policy and expert, batched lockstep episodes.
- CLI: `eval <driver>` (verdict, exit code), `show` (pygame window with the model driving, optional imagined future), video/strip export; Run directory `runs/drive/<date>-<name>/` (config, per-episode records, metrics.json, stdout).
- CPU tests with tiny/untrained Policies and the expert (parity with Step 2), runtime < 60 s; manual acceptance steps.

Check: module/API/CLI listing, output files, test list, manual acceptance procedure.
