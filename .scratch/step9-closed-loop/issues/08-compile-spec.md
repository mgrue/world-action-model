# Compile SPEC.md

Labels: wayfinder:task
Status: open
Assignee:
Blocked by: 01-does-it-drive.md, 02-chunk-execution-practice.md, 03-replanning-scheme.md, 04-episode-protocol.md, 05-metrics-and-gates.md, 06-robustness-report.md, 07-api-cli-tests.md
Parent: ../MAP.md

## Question

Assemble `SPEC.md` from the closed tickets, with a clarifications section for gaps filled while compiling.

Check: `SPEC.md` covers every closed ticket and the PLAN Step 9 check; an agent could implement it without follow-up questions.

## Comments
- All decision tickets (01–07) are closed; vocabulary added to `CONTEXT.md` (Evaluation set, Fractional laps, Fork flip, Saboteur, Stress condition, Push, Driver, Scheme). Prototypes to cite as references: `prototypes/replan_variants.py` (Scheme), `prototypes/gate_saboteurs.py` (gates), `prototypes/stress_calibrate.py` (Stress conditions). Deliverables include the explainer `docs/step9-closed-loop.md`.
