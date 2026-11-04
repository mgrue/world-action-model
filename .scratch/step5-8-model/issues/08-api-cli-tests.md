# Package layout, API, CLI, tests and acceptance

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 03-transformer-blocks.md, 05-training-recipe.md, 06-sampler.md, 07-evaluation-and-checks.md
Parent: ../MAP.md

## Question

What modules, API and commands deliver the model, and what proves them correct?

- Modules in `src/wam/` by concern (`model.py`, `train.py`, `sample.py`, `evaluate.py`?), reuse of Step 4 helpers (Runs, promote/load, provenance).
- Policy API for Step 9: `load_policy(name)` → `act(frame, state, instruction, *, seed) -> actions (16, 2)` (+ optional imagined Frames); batched variant for Step 10.
- Checkpoint `checkpoints/wam-v1/` with card (config, compressor name `vae-v1`, dataset manifest hash, metrics).
- CLI: `train`, `eval`, `sample` / `show`, `promote` — mirroring `wam.compressor`.
- Automated CPU tests (tiny model, tiny dataset): token packing shapes and positions, noising formula, loss masks per Step 10 switch, determinism, resume, sampler shapes, API round trip.
- Manual acceptance: the PLAN checks for Steps 7 and 8.

Check: API and CLI sketch with signatures, the list of automated tests, and the manual acceptance steps.

## Comments
