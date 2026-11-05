# Metrics and gates

Labels: wayfinder:grilling
Status: closed
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

## Resolution

**Metrics and gates** (grilled with mgrue, 2026-11-05).

**Gates** (absolute, set from what a good driver does; verdict for `wam-v1` only; PASS iff all four pass; computed on the full 120-episode gate set at the default scheme; each rate printed with its 95 % Wilson interval):

| Gate | Definition | Threshold |
|---|---|---|
| Crashes | `crash_rate` | ≤ 5 % (≤ 6 / 120) |
| Laps | mean **Fractional laps** = `progress_m / track.length_m` | ≥ 3.5 (expert 4.12; ≈ 85 % of expert pace when crash-free) |
| Instruction followed | `instruction_followed_rate` (every scored Passage correct, no wrong-way Ticks) | ≥ 95 % |
| `any` balance | `outer_fraction` over scored `any` Passages | 0.30 – 0.70; `None` (no scored `any` Passage) = FAIL |

Integer `laps` was rejected as the gate: it is `floor(progress / 88.06 m)`, so a crash-free driver at 97 % of expert pace scores 3 everywhere and would fail ≥ 3.5.

**Report-only** (same table for every driver, rows per combination + total; expert row as reference): `success_rate`, integer `laps_mean`/`laps_min`, Fractional laps of non-crashed episodes, mean lap time; per-episode Crash Tick + Station, fork section or not, Station histogram; distance to the nearest Route centreline per Tick (mean, p95, max); steer RMSE of executed action vs `Expert.control` on visited States, binned by distance off line (< 0.3 / 0.3–0.6 / 0.6–1 / > 1 m); wrong Passages, Passages per Branch per combination, Island Crashes, **Fork flips** (Policies only; definition from [Replanning scheme](03-replanning-scheme.md)); mean |Δsteer| per Tick, wrong-way Ticks; ms per replan, wall time. Flips stay report-only and best of 4 stays the default (no flip caused an Island Crash; their effects are gated via followed + balance).

**Drivers and seeds:**
- `eval <driver>`: default scheme on the gate set. `wam-v1` → verdict + exit code; expert (clean, `burst_peak = 0` — the upper bound) and Step 10 variants → same report and gate table marked "not gated". Any override (`--execute-k`, `--cfg-action`, `--n-samples`, `--n-steps`, …) or `--episodes` < 120 → "non-default scheme" / "not a gate run", no verdict.
- `eval <driver> --sweep <name>`: dev seeds `CLOSED_LOOP_SEED_BLOCK + 10_000 + k`, 60 episodes per point, one report row per point, never a verdict. Grids: `replan` execute_k {1, 2, 4, 8, 16} (≈ 40 min), `cfg` cfg_action {1, 1.5, 2, 3} (≈ 1.5–2 h), `samples` n_samples {1, 4, 8}, `steps` n_steps {2, 4, 10}.

**Validation — Saboteurs** (the clean expert with one defect, same closed loop, 120 gate seeds; `prototypes/gate_saboteurs.py`, 70 s CPU):

| Driver | Crash [95 % CI] | Fractional laps | Followed | Outer | Verdict |
|---|---|---|---|---|---|
| expert (clean) | 0 % [0–3] | 4.12 | 100 % | 0.43 | PASS |
| `zero-steer` (steer 0, expert throttle) | 100 % | 0.05 | 100 % | — | FAIL crash, laps, balance |
| `averaged-fork` (`any`: mean of outer/inner `control`) | 33 % (all 40 `any`) | 2.89 | 100 % | — | FAIL crash, laps, balance |
| `slow` (`v_cap` 5) | 0 % | 3.39 | 100 % | 0.43 | FAIL laps |
| `slow` (`v_cap` 6 / 7) | 0 % | 3.85 / 3.99 | 100 % | 0.44 / 0.43 | PASS (85–97 % pace is fine) |
| `wrong-branch` (told the opposite Branch) | 0 % | 4.12 | 33 % | 0.43 | FAIL followed |
| `always-outer` (`p_outer = 1`) | 0 % | 4.12 | 100 % | 1.00 | FAIL balance |

The spec carries this as an acceptance run, and as CPU tests on gate-set prefixes (< 60 s) asserting each Saboteur fails exactly its gate(s) and the clean expert passes. Note: the expert's outer share on these 120 seeds is 0.43 (coin noise over ≈ 140 Passages), inside 0.30–0.70 with margin.

Assets: [`prototypes/gate_saboteurs.py`](../prototypes/gate_saboteurs.py); outputs [`prototypes/out/gate_saboteurs_stdout.txt`](../prototypes/out/gate_saboteurs_stdout.txt), `gate_saboteurs.json`.
