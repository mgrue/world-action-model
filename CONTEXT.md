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
