# Robustness report

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 05-metrics-and-gates.md
Parent: ../MAP.md

## Question

Which report-only stress tests run, and how?

- Colour Probes (`brown_grass`, `light_road`, `blue_car`, `no_kerb`) and the `wide_fork` Track.
- Starts away from the expert's distribution (lateral offset, heading error, speed), and recovery after a push (state perturbation mid-episode).
- Same metrics as 05; episode counts; whether any of it is ever a gate (default: no).

Check: the list of stress conditions, how each is produced through `CarEnv`/`RenderParams`, counts and runtime.
