# Window definition and time alignment

Labels: wayfinder:grilling
Status: closed
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

## Comments
- Q1 alignment → window at tick t = observed Frame fₜ and State sₜ (both tick t), actions aₜ … aₜ₊₁₅, future Frames fₜ₊₁ … fₜ₊₁₆, Instruction. **Action k ↔ future Frame k = fₜ₊₁₊ₖ** (the Frame rendered after action k is applied), k = 0…15: each action sits on the time step of the Frame it causes (FLUX RoPE convention), and Step 9 replanning uses the same indexing. A window needs States sₜ … sₜ₊₁₆, so t ∈ [0, T−16] (285 positions at T = 300). Frames carry no speed cue; speed comes only from the State token.
- Q2 targets and filter → action targets are always **`clean`**. **Clean-future filter on by default:** a window at t is used only if none of aₜ … aₜ₊₁₅ is perturbed, so `executed == clean` over the future and the action targets agree exactly with the future Frames. The observed tick may be off line after an earlier burst (that's the recovery data). Windows never cross the episode end (dropped, never padded). Each window also carries `executed` (16 × 2) and `perturbed` (16) for analysis; the filter is a loader option (default on), not a separate dataset. About 163 usable windows per 300-tick episode (about 57%).
- Q3 stride and enumeration → **stride 1**: every valid start tick is a window. The loader builds an explicit index of valid (episode, t) per Split on open; shuffling is the training code's job. **Fixed evaluation subsets:** a seeded 2048-window subset per val/test, spread evenly over episodes and Instruction × Direction, so every model and run reports on the same windows (a full pass over about 98k windows would take about an hour at inference speed). The Step 8 check uses the test subset.
- Q4 horizon vs Step 6 token budget → PLAN's **16 actions / 16 future Frames is the default contract**, but the window shape is a **loader setting**, not baked into the recording (data is per-tick States). `horizon` (default 16) sets future actions and Frames; the clean-future filter and index follow it. `future_frames` (default all) selects which future Frames to render, e.g. `(3, 7, 11, 15)`; unrequested Frames are never rendered; actions always cover the full horizon. Step 6 picks its token budget (fewer future Frames, smaller latents, shorter horizon) without re-recording; Step 10's no-video-loss variant can request none. The PLAN check plots the full 16/16 window.
- Q5 metadata → **per tick** (stored with State and actions): `station_m` float32, `branch` int8 (−1 outside the fork section, 0 outer, 1 inner), `planned_branch` int8, plus `perturbed`, `clean`, `executed`. **Per episode:** split, k, seed, Instruction (string and id), Direction, start route, length, termination, `RenderParams`, `TrackParams`, Frame hash. **Per window** (derived by the loader): episode, t, Instruction id, Direction, **`approaches_section`** (outside the fork section at t and entering it within the horizon: the branch-decision windows used to score the Step 10 regression baseline) and **`in_section`** (inside at t). Everything else is derivable from the per-tick fields.
- Q6 layouts → plain numpy, simulator layout, no model normalisation: `obs_frame` (64, 64, 3) uint8; `future_frames` (F, 64, 64, 3) uint8; `state` (5,) float32 State features; `state_raw` (4,) float64 (x, y, θ, v); `actions` (H, 2) float32 `clean`, columns (steer, throttle); `executed` (H, 2) float32; `perturbed` (H,) bool; `future_index` (F,) int64 (the k of each rendered future Frame); `instruction` () int64 id; plus the Q5 metadata scalars. Conversion to CHW float happens model-side (Step 4) on the GPU; core has no torch dependency.

**Resolution (2026-11-03):** a **Window** at tick t of an episode with T ticks, horizon H (default 16) and future Frame selection `future_frames` ⊆ {0…H−1} (default all):

| Field | Definition | Shape / dtype |
|---|---|---|
| `obs_frame` | render(sₜ, episode RenderParams) | (64, 64, 3) uint8 |
| `state` / `state_raw` | features(sₜ) / (x, y, θ, v) of sₜ | (5,) float32 / (4,) float64 |
| `actions` | `clean` aₜ … aₜ₊ₕ₋₁ | (H, 2) float32 |
| `executed`, `perturbed` | same ticks | (H, 2) float32, (H,) bool |
| `future_frames` | render(sₜ₊₁₊ₖ) for k in `future_frames` | (F, 64, 64, 3) uint8 |
| `future_index` | those k | (F,) int64 |
| `instruction` | `INSTRUCTIONS.index(instruction)` | () int64 |
| metadata | episode, split, t, direction, `approaches_section`, `in_section` | scalars |

- **Alignment:** action k ↔ future Frame k = fₜ₊₁₊ₖ (the Frame rendered after action k).
- **Valid t:** 0 ≤ t ≤ T − H; with the default clean-future filter, additionally no perturbed tick in aₜ … aₜ₊ₕ₋₁. Stride 1; the loader builds the (episode, t) index per Split on open.
- **Evaluation subsets:** a seeded 2048-window subset each for val and test, spread evenly over episodes and Instruction × Direction.
- **Per-tick stored fields** (beyond State): `clean`, `executed`, `perturbed`, `station_m`, `branch`, `planned_branch`.
- **Expected count** at T = 300, H = 16, default noise: 285 positions, about 163 usable per episode (about 57%); about 0.98M windows total (train about 0.78M, val/test about 98k each).
- Frame history stays at 1 observed Frame (PLAN). Adding history later is a loader option, since Frames are re-rendered from stored States; deferred to Step 6.
