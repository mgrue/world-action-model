# What happens when the car leaves the road

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: 02-track-car-render-look.md, 03-physics-model.md
Parent: ../MAP.md

## Question

What does the simulator do when the car leaves the road: hard walls (collision stop/bounce), grass slowdown, episode termination, or a combination? How is a "crash" defined precisely (Step 9 counts crashes), and is it reported as an event in info, a terminal flag, or both? Does keyboard mode reset on crash? Note the Island between the branches is off-road too — driving onto it is what a regression baseline is expected to do at the Fork.

Check: the spec defines off-road and crash as predicates over state + track geometry.

## Comments
