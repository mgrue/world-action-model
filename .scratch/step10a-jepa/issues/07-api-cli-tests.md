# Package layout, API, CLI and tests

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 04-check-and-gate.md, 05-training-recipe-and-freezing.md, 06-interface-contract.md
Parent: ../MAP.md

## Question

Where does the code go and how is it driven and tested?

- Modules under `src/wam/` (e.g. `jepa.py`, a shared `probe.py`), and what moves out of `compressor.py` into shared helpers.
- CLI `python -m wam.jepa train | eval | promote` (and a `show` that makes sense without a decoder, e.g. probe errors or feature PCA images?), Run dir layout under `runs/jepa/…`, outputs (`log.csv`, `metrics.json`, …).
- CPU test list with tiny models (< 60 s), and the manual acceptance steps (full training, check, promotion, README / CONTEXT.md updates).

Check: the module list, CLI with exit codes, Run and Checkpoint file lists, test list, acceptance steps.

## Comments
