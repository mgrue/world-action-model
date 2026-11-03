# Physics model and action/state definitions

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 02-track-car-render-look.md
Parent: ../MAP.md

## Question

Which vehicle model (kinematic bicycle vs unicycle, with or without steering/throttle inertia), tick rate, units, action ranges and semantics (steer, throttle — bounded `[-1,1]`? braking/reverse?), max speed and turning radius relative to the chosen track, and exact state vector (position, heading, speed — anything else?) should the simulator use? How is heading represented in state (angle vs sin/cos)? Integration scheme and dtype rules that guarantee determinism.

Constraint from the track ticket: the tightest curve is the 4.88 m branch arc, so the minimum turning radius must sit comfortably below it; track units are metres, y down, heading clockwise-positive.

Check: the spec states the step equations explicitly enough that two independent implementations would match.

## Comments
