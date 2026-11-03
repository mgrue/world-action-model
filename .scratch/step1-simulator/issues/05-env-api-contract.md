# Env API and interface contract

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-python-package-compatibility.md, 03-physics-model.md, 04-off-track-semantics.md
Parent: ../MAP.md

## Question

What is the exact public API Steps 2–4 build on: plain class vs gymnasium `Env`; `reset(seed, options)` and `step(action)` signatures and return values; what lives in State vs info (State = (x,y,θ,v) and State features are fixed by the physics ticket); headless `render()` returning a `uint8 (64,64,3)` array; start-pose randomisation (position along a Route, direction, initial speed) driven by the seed; track queries the expert needs (per-Route centerline, closest point, curvature — the Track has two Routes, outer and inner, see CONTEXT.md); progress along which Route when the car is between Fork and Merge, direction-aware lap counting and wrong-way detection; episode length/truncation (a stopped car is ended only by this); field names for the terminal flag + reason "crash" fixed by the off-road ticket; start poses must not be in a Crash state; where visual-randomisation parameters enter.

Check: the spec contains typed signatures and the determinism tests fixed by the physics ticket (same-machine hash equality + golden trajectory at 1e-9).

## Comments
