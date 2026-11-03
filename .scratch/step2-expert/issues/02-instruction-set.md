# Instruction set and branch choice

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

Which Instructions does the expert follow, and how do they create situations where **two different actions are both correct** (the reason for flow matching, PLAN Step 2 and the Step 10 regression-baseline experiment)?

- Vocabulary: direction only (cw / ccw), branch only (outer / inner), the cross product (4), and/or instructions that leave the branch open ("drive cw, either branch")?
- When the branch is left open, how does the expert choose: seeded random per Fork passage, fixed per episode, alternating? When is the choice made (episode start, N m before the Fork/Merge) — i.e. how long before the split is the choice invisible in the State and Frame?
- Representation: string labels vs enum vs integer id; how Step 3 stores them.
- Does every start pose (any Station, either Route, the episode Direction from `reset`) map to a valid Instruction, or does the Instruction constrain `StartSpec` (e.g. direction must match)?

Check: a table of Instructions with, for each, the required Direction, branch rule, and an example situation where the expert's next 16 actions are bimodal.

## Comments
