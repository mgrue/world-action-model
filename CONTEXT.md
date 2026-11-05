# Toy world action model

A learning project: a tiny simulated car world and a toy FLUX-3-Action-style model that learns to drive it from demonstrations.

## Simulated world

**Track**:
The fixed drivable world: two Routes and the road surface around them. There is exactly one Track.
_Avoid_: map, circuit, level

**Route**:
One closed centerline through the Track, made of straight segments and circular arcs. There are two: the **outer route** and the **inner route**, identical except between the Fork and the Merge.
_Avoid_: path, lane, line

**Fork**:
The point where the outer and inner Routes split. **Merge** is the point where they rejoin.
_Avoid_: junction, split

**Branch**:
The part of a Route between the Fork and the Merge; the outer branch bulges outward, the inner branch into the infield.

**Island**:
The off-road area enclosed between the two Branches.

**Road surface**:
Every point within half the road width of either Route's centerline.
_Avoid_: track (when meaning the asphalt)

**Track units**:
Metres in a fixed 40 m × 40 m world, origin top-left, x right, y down; resolution-independent — pixels exist only after rendering.
_Avoid_: pixels (for world positions)

**Frame**:
One rendered 64×64 RGB image of the whole Track from the fixed overhead camera.
_Avoid_: screenshot, observation (when meaning the image alone)

## Car and control

**State**:
The car's canonical physical state (x, y, θ, v): position in Track units, heading angle, speed. Fully determines the next State given an Action.
_Avoid_: observation, pose (when speed is included)

**State features**:
The model-facing encoding of a State: (x/40, y/40, cos θ, sin θ, v/10).
_Avoid_: state vector (ambiguous)

**Action**:
One tick's command [steer, throttle], each in [-1, 1].
_Avoid_: control, input, command

**Tick**:
One simulation step at 15 Hz: one Action applied, one State update, one Frame rendered.
_Avoid_: frame (when meaning time), step (when ambiguous)

**Crash**:
The event of any car corner leaving the road surface plus kerb (more than 2.8 m from both Routes); it ends the episode.
_Avoid_: collision, off-track (as a noun)

## Episode bookkeeping

**Station**:
Distance in metres along a Route from the Fork, increasing clockwise, in [0, Route length); identical on both Routes outside the Branches.
_Avoid_: arc length, position (for this scalar)

**Direction**:
The travel sense an episode is meant to follow: clockwise (cw, Station increasing) or counterclockwise (ccw).
_Avoid_: orientation, heading (heading is the car's angle)

**Progress**:
Distance accumulated along Station in the episode's Direction; negative when driving the wrong way.

**Lap**:
One full Route length (88 m) of accumulated Progress.

## Expert and instructions

**Instruction**:
The per-episode command the driver follows: `outer`, `inner` or `any`. It picks which Branch to take at every Passage. It does not include Direction, which comes from the start pose.
_Avoid_: command, task, goal

**Null Instruction**:
The model-side placeholder `null` that stands for "Instruction withheld": swapped in for a fraction of training Windows and used for the unconditioned pass of classifier-free guidance. Never in the data; distinct from `any`, which is a real Instruction.
_Avoid_: empty instruction, `any` (when meaning withheld)

**Passage**:
One traversal of the fork section, from entering it at one end (the Fork or the Merge) to leaving it at the other.
_Avoid_: fork crossing, branch visit

**Planned branch**:
The Branch the expert intends to take on its current or next Passage: the Instruction itself for `outer`/`inner`, or a fair coin for `any`, drawn at reset and on each exit from the fork section.

**Expert**:
The scripted driver that follows an Instruction using privileged State and Info (never Frames). It produces a clean Action (the label) and an executed Action (clean plus noise).
_Avoid_: oracle, teacher, autopilot

**Burst**:
A short triangular steering perturbation the Expert adds to the executed Action (never to the label) so the data shows recoveries.
_Avoid_: noise (when meaning one perturbation event), jitter

## Dataset

**Split**:
One of the disjoint episode sets `train`, `val`, `test`, each recorded from its own seed block. Episodes are assigned to Splits whole; windows of one episode never appear in two Splits.
_Avoid_: fold, partition

**Window**:
One training example cut from an episode at tick t: the observed Frame and State at t, the Instruction, the next H expert actions (clean labels) and the H Frames they cause (action k ↔ future Frame k). H = 16 by default.
_Avoid_: sample, clip, chunk (chunk = the predicted action sequence)

**Probe**:
A named world change reserved for Step 10 generalisation tests (a colour or track-shape change outside the training jitter); never used in training data.
_Avoid_: variant, test set (a Probe is a world change, not a Split)

## Model

**Latent**:
The compressor's 8×8 grid of 16-channel values for one Frame (normalised per channel), the unit the world action model reads and predicts instead of pixels.
_Avoid_: embedding, code, feature map

**Checkpoint**:
A promoted, frozen set of trained weights with its provenance card, stored under `checkpoints/<name>/` and referred to by name (e.g. `vae-v1`); never overwritten.
_Avoid_: model file, weights (when meaning the named, frozen artefact)

**Run**:
One training run's scratch directory under `runs/` (config, logs, metrics, contact sheet, final weights); only a Checkpoint promoted from it is used by later Steps.
_Avoid_: experiment, job

**Token**:
One d-dimensional vector in the world action model's input sequence: a 2×2 patch of one Frame's Latent (64 values), the State, the Instruction, or one Action.
_Avoid_: patch (when meaning the projected vector), embedding

**Stream**:
All Tokens of one kind in a Window's sequence, sharing an input projection and (if predicted) an output head: observed Frame, State, Instruction, future Frame, Action.
_Avoid_: modality, channel

**Chunk**:
The sequence of H = 16 Actions the model predicts for one Window (the action Stream's output); Step 9 executes its first few and replans.
_Avoid_: trajectory, plan, window

**Noise level**:
The flow-matching time t ∈ [0, 1] of a Token: x_t = t·noise + (1 − t)·clean, so 0 is clean and 1 is pure noise. One per Window for the predicted Streams; always 0 for the conditioning Streams.
_Avoid_: timestep (when meaning the value), sigma, diffusion step


## Closed loop

**Evaluation set**:
One of the disjoint closed-loop episode lists `dev`, `gate`, `robustness`, each a fixed seed range inside `CLOSED_LOOP_SEED_BLOCK`; episode k of a set plays Instruction × Direction combination k mod 6. Every driver runs the same list. Only the `gate` set produces a verdict, and it is never used for tuning.
_Avoid_: Split (that is recorded data), test set, validation set

**Fractional laps**:
`progress_m / track.length_m` at the end of an episode — the Lap count without the floor. The Step 9 laps gate uses it; the integer Lap count stays in reports.
_Avoid_: laps (when meaning the fractional value)

**Fork flip**:
A Passage approach where the replans in the 1–12 Ticks before fork entry select Chunks that classify (against the Fork references) to different Branches. Report-only in Step 9.
_Avoid_: dithering, wobble

**Saboteur**:
A deliberately broken driver (the clean Expert with one defect) used to prove a gate fails when it should.
_Avoid_: baseline, negative control (when meaning these drivers)
