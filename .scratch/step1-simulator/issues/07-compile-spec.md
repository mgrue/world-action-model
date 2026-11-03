# Compile SPEC.md

Labels: wayfinder:task
Status: closed
Assignee: mgrue
Blocked by: 02-track-car-render-look.md, 03-physics-model.md, 04-off-track-semantics.md, 05-env-api-contract.md, 06-keyboard-driving-mode.md
Parent: ../MAP.md

## Question

Assemble the resolved decisions into `.scratch/step1-simulator/SPEC.md` (AFK): module layout, physics equations, track definition, rendering rules, API signatures, keyboard mode, acceptance tests. Before starting, confirm the map's Not-yet-specified section is empty and no frontier tickets remain open.

## Comments

**Resolution (2026-11-03):** SPEC.md written from the resolutions of all six decision tickets plus CONTEXT.md. Not-yet-specified was empty and no other tickets were open. Derived values computed exactly: α = 67.380°, ρ = 4.875 m, branch length 22.932 m, Route length 88.056 m. Gaps filled while compiling are listed in SPEC.md §11 "Clarifications made while compiling" (reference point = rectangle centre, exact computed lengths, tie-breaks, Station unwrapping, fixed RNG draw order, reset retry limit, raise after truncation, recording reasons, render performance target).
