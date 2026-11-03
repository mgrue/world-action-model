# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 01-storage-and-loading.md, 02-episode-mix-and-splits.md, 03-visual-randomisation.md, 04-window-definition.md, 05-recorder-loader-and-plot.md
Parent: ../MAP.md

## Question

Assemble the resolved decisions into `.scratch/step3-dataset/SPEC.md` (AFK): recorded set and splits, randomisation ranges and held-out probes, storage format and provenance metadata, window definition and filter, recorder/loader API, CLI, automated tests and a short manual acceptance checklist (including the PLAN check: a plotted window whose frames and actions line up). Before starting, confirm the map's Not-yet-specified section is empty and no frontier tickets remain open.

## Comments
**Resolution (2026-11-03):** [SPEC.md](../SPEC.md) written from the resolutions of tickets 01–05, `research/01-storage-and-loading.md` and the prototypes (`storage_bench.py`, `recolour_render.py`, `visual_contact_sheet.py`). Not-yet-specified was empty and no other tickets were open. The gaps filled are listed in SPEC.md §14: exported seed/Split constants, `start_route` stores the requested route, `planned_branch` read after `act`, jitter draw order, one Frame hash per episode, `Probe` dataclass with `apply`, Window dict extras, eval-subset algorithm (`[seed, 4]` quotas), hidden CLI size options, `env.py` in source hashes, fixed hash-check episodes, `verify` also re-simulates, `any_outer_fraction` uses the Step 2 passage rule, and derived performance targets.
