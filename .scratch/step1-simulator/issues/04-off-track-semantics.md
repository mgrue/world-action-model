# What happens when the car leaves the road

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 02-track-car-render-look.md, 03-physics-model.md
Parent: ../MAP.md

## Question

What does the simulator do when the car leaves the road: hard walls (collision stop/bounce), grass slowdown, episode termination, or a combination? How is a "crash" defined precisely (Step 9 counts crashes), and is it reported as an event in info, a terminal flag, or both? Does keyboard mode reset on crash? Physics context: too much speed makes the steering clamp understeer the car wide; there is no reverse, so a car stopped off-road can't back out. Note the Island between the branches is off-road too — driving onto it is what a regression baseline is expected to do at the Fork.

Check: the spec defines off-road and crash as predicates over state + track geometry.

## Comments

- Q1 behaviour → **leaving the road is a Crash and ends the episode immediately**; no walls, no grass physics; physics unchanged.
- Q2 predicate → **Crash ⇔ any of the 4 car corners is > 2.8 m (kerb outer edge) from the nearest point on either Route centerline**, evaluated after each tick's State update. Pure function of State + Track geometry; union of both Routes is drivable at the Fork.
- Q3 reporting → the crashing tick returns its State and Frame (car touching grass) plus a terminal flag and reason "crash"; step() after a crash without reset() raises. Not crashes: stopping on road (ended by step limit), wrong-way driving (Env API). Field names/signature shape belong to the Env API ticket; keyboard-mode presentation to the Keyboard ticket.

**Resolution (2026-11-03):**
- **Behaviour:** leaving the road is a **Crash** and terminates the episode; no walls, no grass physics, physics unchanged.
- **Predicate:** Crash ⇔ any of the 4 car corners (car 4.4 × 2.5 m rectangle at (x, y, θ)) lies > 2.8 m from the nearest point on either Route centerline; evaluated after each State update.
- **Reporting:** crashing tick returns State + Frame, terminated = true, termination reason "crash"; further step() without reset() raises an error.
- **Non-crashes:** standing still on road, wrong-way driving.
