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
