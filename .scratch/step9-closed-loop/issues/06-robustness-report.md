# Robustness report

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 05-metrics-and-gates.md
Parent: ../MAP.md

## Question

Which report-only stress tests run, and how?

- Colour Probes (`brown_grass`, `light_road`, `blue_car`, `no_kerb`) and the `wide_fork` Track.
- Starts away from the expert's distribution (lateral offset, heading error, speed), and recovery after a push (state perturbation mid-episode).
- Same metrics as 05; episode counts; whether any of it is ever a gate (default: no).

Check: the list of stress conditions, how each is produced through `CarEnv`/`RenderParams`, counts and runtime.

## Comments
- From 04: robustness episodes use seeds `CLOSED_LOOP_SEED_BLOCK + 200_000 + k` (combination `k % 6`), same episode rule as the gate set (900 Ticks, Step 3 start rule) except the stressed factor. Budget reference: ≈ 23 s per episode for the default scheme → ≈ 46 min per 120 episodes; 60 per condition gives ±≈ 11 pp at 25 %.
- From 05: report the same metric table as the gate report (gates shown as "report-only", never a verdict), incl. Fractional laps, Crash Stations, distance off the centreline and action error vs `Expert.control` binned by distance — the last two are the natural readouts for off-distribution starts and pushes. Run the clean expert under every stress condition as the reference row (it must stay crash-free under colour Probes trivially; under pushes/offset starts it shows what is recoverable at all).
