# Track geometry, car size and look at 64×64

Labels: wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What track layout, road and lane width, car size, nose marker and colour palette make a 64×64 overhead frame of the **whole track** readable — car position and heading clearly visible, two lanes distinguishable, both driving directions possible — while the car still occupies enough pixels to survive the 8×8 latent grid of Step 4 (8 px per latent cell)? Includes: track shape family (oval / rounded rectangle / spline), centerline representation, world-to-pixel scale, and whether to anti-alias/supersample so sub-pixel motion is visible.

Prototype: render a few candidate layouts at 64×64 (and upscaled ×8) with the car at several poses, for the human to react to.

Check: the human picks a candidate and can read the car's heading at native 64×64.

## Comments
