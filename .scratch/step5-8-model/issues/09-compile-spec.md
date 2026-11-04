# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 01-throughput-budget.md, 02-tokens-and-positions.md, 03-transformer-blocks.md, 04-flow-matching-objective.md, 05-training-recipe.md, 06-sampler.md, 07-evaluation-and-checks.md, 08-api-cli-tests.md
Parent: ../MAP.md

## Question

Assemble the resolved decisions into `.scratch/step5-8-model/SPEC.md` (AFK): scope and layout, environment, token layout and Instruction encoding, transformer, flow-matching objective, training recipe, sampler, evaluation checks and thresholds, Runs and Checkpoints, API for Steps 9–10, CLI, automated tests, manual acceptance (PLAN Steps 7 and 8 checks), hand-offs and out of scope. Before starting, confirm the map's Not-yet-specified section is empty and no frontier tickets remain open.

Check: `SPEC.md` exists and covers every closed ticket.

## Comments
- Pre-check: the map's Not-yet-specified section is empty; tickets 01–08 are closed.
- Verified while compiling: `windows(future_frames=())` → `(0, 64, 64, 3)`, `encode_windows` → `(B, 0, 16, 8, 8)`; `windows(horizon=8, future_frames=range(1, 8, 2))` works — the no-video and short-H variants need no Step 3 change.

**Resolution (2026-11-04):** [`SPEC.md`](../SPEC.md) compiled from tickets 01–08: scope and layout, environment and budget, token layout / positions / Instruction encoding, transformer, flow-matching objective and switches, training recipe and freezing, sampler, Policy API for Steps 9–10, evaluation gates and report, Runs and Checkpoints, CLI, 21 automated tests plus two conditional ones, performance targets, manual acceptance (PLAN Steps 7 and 8), hand-offs, 12 clarifications made while compiling, out of scope.

