# Keyboard driving mode

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 01-python-package-compatibility.md, 03-physics-model.md
Parent: ../MAP.md

## Question

How does a human drive the car: key mapping, how digital key presses map to continuous steer/throttle (instant vs ramped; note throttle 0 holds speed, negative throttle brakes, no reverse), real-time loop at the sim tick rate, upscaled display of the 64×64 frame (and optionally a debug overlay with state/progress), reset/quit keys, what the human sees on a Crash (episode terminates; e.g. pause + auto-reset), and whether a drive can be recorded (seed + StartSpec + action log, using CarEnv from the Env API ticket) and replayed to prove determinism?

Check: the spec defines the PLAN.md Step 1 check ("a human can drive the car and the rendering looks right") as a concrete manual test script.

## Comments
