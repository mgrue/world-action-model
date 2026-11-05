# Metrics and gates

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-replanning-scheme.md, 04-episode-protocol.md
Parent: ../MAP.md

## Question

What does the report contain, and what must `wam-v1` reach to PASS?

- Reuse `score_episode` / `summarize` (crash, laps, Instruction followed, success, `any` outer fraction); additions: closed-loop Fork metrics (Island entries, flip-flops, Passages per branch), lap time, distance off the Route centreline, steer smoothness, wrong-way.
- Gates for `wam-v1` vs report-only; thresholds absolute or relative to the expert row; how CFG / scheme sweeps are reported.
- Validation of the metrics: the expert passes; a deliberately bad driver (e.g. zero steer, averaged Fork) fails.

Check: a metric/gate table with thresholds and the run that validates them.

## Comments
- From 01: Step 9 is evaluation only — set gates from what a good driver must achieve (expert row as reference), not from `wam-v1`'s numbers; `wam-v1` failing is an accepted outcome. Useful diagnostics seen in 01: distance off the centreline (mean / p95), Crash location (Station, fork section or not), and executed action vs `Expert.control` on visited States (the off-distribution action error) — consider them for the report.
