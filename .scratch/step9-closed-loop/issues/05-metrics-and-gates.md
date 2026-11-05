# Metrics and gates

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
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
- From 03: Fork flip metric used in the prototype — per Passage approach, classify every replan's selected Chunk 1–12 Ticks before entry against `evaluate.fork_references` (Direction-normalised steer), flip if they disagree; default scheme ≈ 14/32 `any` approaches flip, never into the Island. Decide whether flips are a gate, report-only, or justify best of 8 (≈ 2× cost).
- From 04: gate set = 120 episodes (20 per Instruction × Direction, 40 `any`), seeds `CLOSED_LOOP_SEED_BLOCK + 100_000 + k`. 95 % Wilson half-widths on rates at 120: 0 % → ≤ 3.1 %, 5 % → 2.3–10.5 %, 25 % → 18.1–33.4 % — pick thresholds this count resolves (e.g. "crash ≤ 5 %" is resolvable against 25 %, "≤ 1 %" is not). Report CIs next to the headline rates.
