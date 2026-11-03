# Env API and interface contract

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 01-python-package-compatibility.md, 03-physics-model.md, 04-off-track-semantics.md
Parent: ../MAP.md

## Question

What is the exact public API Steps 2–4 build on: plain class vs gymnasium `Env`; `reset(seed, options)` and `step(action)` signatures and return values; what lives in state vs info; headless `render()` returning a `uint8 (64,64,3)` array; start-pose randomisation (position along centerline, direction, lane, initial speed) driven by the seed; track queries the expert needs (centerline, lane offset, closest point, curvature); progress along the centerline, direction-aware lap counting and wrong-way detection; episode length/truncation; where visual-randomisation parameters enter.

Check: the spec contains typed signatures and a determinism test (same seed + same action sequence → identical states and frame hashes).

## Comments
