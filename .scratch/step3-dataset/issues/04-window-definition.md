# Window definition and time alignment

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

Pin down a training window exactly, so that "frames and actions line up in time" is checkable and Steps 6–9 inherit one convention.

- Alignment: with observed Frame f_t (rendered from State s_t), is the window {f_t, s_t, a_t … a_{t+15}, f_{t+1} … f_{t+16}}, so action k causes future Frame k? Confirm against Step 1's tick definition (one Action → one State update → one Frame) and against Step 9 replanning (execute the first few actions, then replan).
- Which action is the target: `clean` (label) everywhere; is `executed` stored too (for analysis, or to train on noisy futures)?
- Window filter: Step 2 default is "no perturbed tick in the 16-tick future". Is the observed tick allowed to be perturbed? Windows that would cross the episode end are dropped or padded?
- Stride: every tick (heavily overlapping) or subsampled? Is the window set enumerated up front (an index) or sampled at random positions per epoch?
- State token: State features `(x/40, y/40, cos θ, sin θ, v/10)` from CONTEXT.md; actions stay in [−1, 1] raw. Any other normalisation belongs to the model.
- Instruction: stored as the string and its `INSTRUCTIONS` id.
- Per-window metadata useful later: Direction, planned branch, distance to the Fork/Merge, whether the window spans a branch decision (to slice multimodal windows for the Step 10 regression baseline), episode id and tick.

Check: a precise definition (indices, dtypes, shapes) of one window, the filter rule, the stride, the metadata fields, and the expected window count per episode at default noise.
