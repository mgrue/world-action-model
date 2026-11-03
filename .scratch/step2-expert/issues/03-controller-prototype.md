# Expert controller and speed profile

Labels: wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by: 01-controller-research.md
Parent: ../MAP.md

## Question

What exact control law and speed profile let the expert follow either Route in either Direction from every valid start pose, without Crashing, at a sensible pace (not crawling, not on the grip limit)?

- Steering: controller from the research ticket, its gains/lookahead (fixed or speed-scaled), and how it handles the Fork/Merge where both Routes coincide and then split (target Route switches at the section boundary).
- Speed: target speed per Station from curvature (`√(a_lat·R)` with margin), braking horizon using BRAKE = 8 m/s², behaviour from awkward starts (6 m/s already on a branch arc, ±10° heading, ±1 m lateral).
- Smoothness: are the expert's actions smooth enough to be learnable 16 at a time (steer chatter, bang-bang throttle)?

Prototype: a throwaway controller run headless over many seeds × Route × Direction, plus plots/PNGs of trajectories and action traces, and an optional `--show` window for the human to watch.

Check: over ≥ 1000 seeded starts per Route × Direction, zero Crashes in 900-tick episodes, the correct branch on every passage, ≥ 4 laps per episode on average, and the human agrees the driving and action traces look right.

## Comments
