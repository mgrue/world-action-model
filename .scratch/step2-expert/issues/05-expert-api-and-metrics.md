# Expert API, metrics and CLI

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 02-instruction-set.md, 04-expert-variation.md
Parent: ../MAP.md

## Question

What is the interface contract Step 3 (dataset recording) and Step 9 (closed-loop evaluation) build on?

- Module and types: e.g. `toycar.expert` with `Instruction`, `ExpertParams`, `Expert(track, instruction, params, seed)`, `reset(state, info)`, `act(state, info) -> Action`; stateful (branch decision memory) vs pure.
- Privileged inputs: the expert reads State and Info (not Frames) — confirm, and whether it needs anything the Step 1 env doesn't expose (e.g. a per-passage branch history rather than only `last_branch`).
- **Metrics shared with Step 9**: laps completed, Crash rate, instruction followed (Direction held, no wrong-way; correct branch on every Fork passage where one is required; for open-branch instructions, the branch-choice split). Exact definitions and where they live (a reusable episode-scoring function).
- CLI: `python -m toycar.expert` to watch the expert drive (reusing drive.py's window/panel) and to run the headless reliability evaluation with a summary table.

Check: typed signatures, metric definitions precise enough that Step 9 can score a learned policy with the same function, and a CLI usage block.

## Comments
