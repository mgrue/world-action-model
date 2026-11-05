# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 01-does-it-drive.md, 02-chunk-execution-practice.md, 03-replanning-scheme.md, 04-episode-protocol.md, 05-metrics-and-gates.md, 06-robustness-report.md, 07-api-cli-tests.md
Parent: ../MAP.md

## Question

Assemble `SPEC.md` from the closed tickets, with a clarifications section for gaps filled while compiling.

Check: `SPEC.md` covers every closed ticket and the PLAN Step 9 check; an agent could implement it without follow-up questions.

## Comments
- All decision tickets (01–07) are closed; vocabulary added to `CONTEXT.md` (Evaluation set, Fractional laps, Fork flip, Saboteur, Stress condition, Push, Driver, Scheme). Prototypes to cite as references: `prototypes/replan_variants.py` (Scheme), `prototypes/gate_saboteurs.py` (gates), `prototypes/stress_calibrate.py` (Stress conditions). Deliverables include the explainer `docs/step9-closed-loop.md`.

## Resolution

[`SPEC.md`](../SPEC.md) compiled (2026-11-05) from the seven closed tickets: scope/layout (§1), budget (§2), Evaluation sets and episode rule (§3, ticket 04), Driver protocol, Scheme, PolicyDriver noise/selection, Expert and Saboteurs, lockstep `run_episodes` and records (§4, tickets 02/03/05/07), metrics incl. Fork flips (§5), gates + verdict rules + Saboteur validation (§6, ticket 05), sweeps (§7), robustness Stress conditions and calibration (§8, ticket 06), Run directory (§9), CLI (§10), `noise=` API change (§11), tests (§12), performance targets (§13), manual acceptance incl. explainer (§14), hand-offs (§15), 10 compile-time clarifications (§16), out of scope (§17). PLAN Step 9 check mapped to the §6 gates on the §3 gate set with the §4 loop.
