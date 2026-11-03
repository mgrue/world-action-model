# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 01-controller-research.md, 02-instruction-set.md, 03-controller-prototype.md, 04-expert-variation.md, 05-expert-api-and-metrics.md
Parent: ../MAP.md

## Question

Assemble the resolved decisions into `.scratch/step2-expert/SPEC.md` (AFK): Instruction set, controller law and speed profile with exact constants, variation knobs and RNG rules, expert API, shared metrics, CLI, automated tests (reliability over seeds, determinism, branch-choice statistics) and a short manual acceptance checklist. Before starting, confirm the map's Not-yet-specified section is empty and no frontier tickets remain open.

## Comments
**Resolution (2026-11-03):** SPEC.md written from the resolutions of tickets 01–05, `research/01-controller-and-demo-noise.md` and the final prototype (`prototypes/expert_prototype.py`, as last swept). Not-yet-specified was empty and no other tickets were open. Exact rules (call order inside `act`, coin/noise draw orders, feasibility rollout, profile grid, burst triangle) were taken from the prototype; the gaps filled are listed in SPEC.md §11 (act ordering, coin only under `any`, rollout start-state check, profile grid/lookup, burst length 8–22 ticks, no noise draws at peak 0, `track` argument of `score_episode`, `eval` uses the recording rule, exempt flag not in the API, prototype counts not binding).
