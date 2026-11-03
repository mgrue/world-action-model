# Instruction set and branch choice

Labels: wayfinder:grilling
Status: closed
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
- Q1 vocabulary → **three branch-only Instructions: `outer`, `inner`, `any`**. Direction is not part of the Instruction: it remains an episode property from `reset`, and both Directions appear in the data. Why: Direction can be read from the car's heading (starts face the Direction ±10°, there's no reverse, and a U-turn is only possible at the outer bulge), whereas the branch is invisible until the split. `outer`/`inner` test instruction-following and CFG; `any` provides the two-valid-actions case for the regression-vs-flow experiment.
- Q2 branch choice under `any` → **independent fair coin per Fork passage** (50/50) from the expert's own seeded RNG. A single-frame, history-free model sees every approach as two valid choices whichever rule is used. Per passage gives a balanced split and tests whether a policy holds its choice on every passage in Step 9. Alternating is a hidden pattern, so it was rejected; per-episode gives high variance. Whether p = 0.5 becomes a tunable setting is ticket 04.
- Q3 flip timing → the expert always holds a **planned branch**. Under `outer`/`inner` it equals the Instruction. Under `any` it is drawn at `reset` and redrawn each time the car **leaves the fork section** (either end). This is one rule for both Directions, and it draws at fixed events so replays are reproducible. The visible divergence starts when the controller's lookahead reaches the split (ticket 03), not at flip time.
- Q4 starts inside the fork section → **both layers**. (a) Recording rule: for `outer`/`inner` the caller sets `StartSpec(route=instruction)` (harmless outside the section, because the Routes coincide there); `any` keeps a random route. (b) Expert safety rule: if it is in a branch other than the planned one, it follows the branch it is in for that passage, and the passage is **exempt** (not failed) in the metrics. Under `any`, a start inside the section makes the current branch the planned one, and no coin is drawn until it exits.
- Q5 representation → `Instruction = Literal["outer", "inner", "any"]` (same style as `Branch`) plus a fixed, frozen, ordered tuple `INSTRUCTIONS = ("outer", "inner", "any")`; the index is the integer id for Step 5's embedding table, and new Instructions are added only at the end. Step 3 stores the string (source of truth) and may also store the id. Free-text phrasings are deferred to Step 5.
- Q6 mid-episode changes → **fixed per episode** in the expert and Step 3, with Instructions spread evenly across episodes. A history-free model can't tell a changed Instruction from a fixed one, so changes would add nothing to the data. Step 9 may still switch Instructions mid-episode as an evaluation probe.
- Q7 CFG null instruction → **model-side only**: an extra embedding slot at index `len(INSTRUCTIONS)` (= 3), used by training dropout and the unconditional sampling pass. It is not an Instruction: the expert never receives it and Step 3 never records it. Distinct from `any`, a real Instruction; null means "unknown", i.e. the average over all Instructions.

**Resolution (2026-11-03):**

| Instruction | id | Direction | Branch rule | Example where the next 16 actions are bimodal |
|---|---|---|---|---|
| `outer` | 0 | either (from `reset`) | planned branch = outer on every passage | none from the instruction itself: actions at the split are unimodal (bear left in cw, right in ccw), and the label alone decides them |
| `inner` | 1 | either | planned branch = inner on every passage | none from the instruction itself; mirrors `outer`, so it is the contrast the instruction-following / CFG experiments measure |
| `any` | 2 | either | fair coin per passage, drawn at `reset` and on each exit from the fork section | cw: car on the left end arc ≈ 5 m before the Fork at ≈ 6 m/s. The 16-tick chunk (≈ 6.4 m) either turns left and up onto the outer branch or right and down onto the inner one, and the midpoint drives into the Island tip. ccw: the same on the right end arc approaching the Merge |

- Instruction fixed per episode; Direction is not an Instruction.
- Recording starts match the Instruction (`StartSpec(route=instruction)` for `outer`/`inner`); a wrong-branch passage is followed and exempt.
- `INSTRUCTIONS` order is frozen; null for CFG is model-side, index 3.
- Glossary: Instruction, Passage, Planned branch added to CONTEXT.md.
