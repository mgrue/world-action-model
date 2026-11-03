# Recorder, loader API, CLI and window plot

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 01-storage-and-loading.md, 02-episode-mix-and-splits.md, 04-window-definition.md
Parent: ../MAP.md

## Question

What module, API and commands deliver the dataset, and what proves it correct?

- Module layout: `toycar.dataset` (record, load, windows) — no torch dependency in the core; a thin optional PyTorch `Dataset` wrapper, or leave that to Step 4/6?
- Recorder CLI: `python -m toycar.dataset record --out DIR …` with parallel workers; resumable; deterministic regardless of worker count; writes a manifest (versions, params, seed ranges, counts, hashes).
- Loader API: open a dataset, list episodes per split, get a window by index, iterate windows; returns numpy arrays with the shapes fixed by ticket 04.
- Verification: replay check (re-simulate an episode from its metadata and compare States/Frames), dataset stats (Instruction/Direction balance, branch split, perturbed fraction, speed histogram).
- PLAN check tool: `python -m toycar.dataset plot` showing one window — observed Frame, the 16 future Frames in a strip, the action traces (steer/throttle, clean vs executed), with tick indices so alignment is visible. Output PNG (matplotlib as a dev/optional dependency?) or a pygame viewer?

Check: an API and CLI sketch with signatures, the manifest fields, the list of automated tests, and the manual acceptance steps.
