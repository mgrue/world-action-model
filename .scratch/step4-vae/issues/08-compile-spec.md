# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 01-torch-on-mac.md, 02-package-layout-and-checkpoints.md, 03-architecture-and-objective.md, 04-car-visibility-check.md, 05-training-recipe.md, 06-latents-for-windows.md, 07-api-cli-tests.md
Parent: ../MAP.md

## Question

Assemble the resolved decisions into `.scratch/step4-vae/SPEC.md` (AFK): environment and dependencies, package layout, architecture and loss, the car-visibility metric and thresholds, training recipe and freezing, checkpoint format and provenance, latent contract for Steps 6–8, API, CLI, automated tests and a short manual acceptance checklist (including the PLAN check). Before starting, confirm the map's Not-yet-specified section is empty and no frontier tickets remain open.

## Comments
- Pre-check: the map's Not-yet-specified section is empty; tickets 01–07 are closed.

**Resolution (2026-11-04):** [`SPEC.md`](../SPEC.md) compiled from tickets 01–07: scope and layout, environment, network and loss, the car-visibility check (with the ticket 04 amendment), training recipe and freezing, Runs and Checkpoints, the latent contract, API, CLI, 15 automated tests plus two conditional ones, performance targets, manual acceptance (PLAN check), hand-offs, 12 clarifications made while compiling (one additive Step 3 method, `FrameItems.locate`) and out of scope.

