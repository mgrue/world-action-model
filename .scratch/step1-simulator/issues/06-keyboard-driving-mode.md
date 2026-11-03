# Keyboard driving mode

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 01-python-package-compatibility.md, 03-physics-model.md
Parent: ../MAP.md

## Question

How does a human drive the car: key mapping, how digital key presses map to continuous steer/throttle (instant vs ramped; note throttle 0 holds speed, negative throttle brakes, no reverse), real-time loop at the sim tick rate, upscaled display of the 64×64 frame (and optionally a debug overlay with state/progress), reset/quit keys, what the human sees on a Crash (episode terminates; e.g. pause + auto-reset), and whether a drive can be recorded (seed + StartSpec + action log, using CarEnv from the Env API ticket) and replayed to prove determinism?

Check: the spec defines the PLAN.md Step 1 check ("a human can drive the car and the rendering looks right") as a concrete manual test script.

## Comments

- Q1 key mapping → ↑/W throttle +1, ↓/S throttle −1 (brake, never reverse), neither → 0 (cruise). ←/A, →/D ramp steer toward ∓1/±1 at 4 /s; no steer key → ramp back to 0 at 6 /s. Ramp lives only in drive.py's input layer; recorded actions are the ramped values.
- Q2 display/loop → pygame-ce; fixed 15 Hz loop (Clock.tick(15)): pump events, key.get_pressed(), one env.step, env.render(). Main view = true 64×64 frame ×8 nearest-neighbour (512×512); H toggles hi-res view render(resolution=512); Tab toggles a debug panel to the right (x, y, θ°, v, steer, throttle, station, progress, laps, direction, branch/last branch, wrong_way, steps, seed). Window ≈ 732×512.
- Q3 lifecycle/CLI → `uv run python -m toycar.drive [--seed N] [--direction cw|ccw] [--route outer|inner] [--speed V] [--record PATH]`; no --seed → clock-based seed shown in panel + terminal; default start speed 0 (rest of pose random). Crash → freeze last frame with red "CRASH" banner 1.5 s, auto-reset with seed+1. max_steps = 10**9. Keys: R reset (seed+1), Shift+R restart same seed, P pause, Esc/Q quit.
- Q4 record/replay → `--record DIR` writes `DIR/drive_<seed>.npz` per episode on end (crash/reset/quit): `meta` JSON (seed, StartSpec, RenderParams, TrackParams, toycar version, termination), `actions` float64 (N,2) as passed to step, `states` float64 (N+1,4), `frame_sha256` (N+1,). `python -m toycar.replay FILE [--show] [--export-png DIR]`: default headless re-run asserting bit-identical States + frame hashes (PASS or first diverging tick); --show plays back at 15 Hz; --export-png dumps frames. Not Step 3's dataset format.
- Q5 acceptance script → agreed (step 6 corrected: relaunch with --direction ccw, since laps only count in the episode Direction).

**Resolution (2026-11-03):**
- **Input:** ↑/W throttle +1, ↓/S −1, none → 0 (cruise). ← / A steer ramps toward −1, → / D toward +1 at 4 /s; released → back to 0 at 6 /s. Ramp only in drive.py; recorded actions are the ramped values.
- **Loop/display:** pygame-ce, fixed 15 Hz (Clock.tick(15)), one env.step + env.render per loop. Main view 64×64 ×8 nearest (512×512); H = hi-res render(resolution=512); Tab = debug panel right of the frame (x, y, θ°, v, steer, throttle, station, progress, laps, direction, branch, last_branch, wrong_way, steps, seed).
- **CLI:** `uv run python -m toycar.drive [--seed N] [--direction cw|ccw] [--route outer|inner] [--speed V] [--record DIR]`; clock seed if omitted (printed + shown); default start speed 0. max_steps = 10**9.
- **Lifecycle:** Crash → freeze last frame + red "CRASH" banner 1.5 s → auto-reset seed+1. R = reset seed+1, Shift+R = restart same seed, P = pause, Esc/Q = quit.
- **Recording:** `DIR/drive_<seed>.npz` per episode: meta JSON (seed, StartSpec, RenderParams, TrackParams, toycar version, termination), actions f64 (N,2), states f64 (N+1,4), frame_sha256 (N+1,).
- **Replay:** `python -m toycar.replay FILE [--show] [--export-png DIR]` — default headless bit-exact check of States + frame hashes (PASS / first diverging tick).
- **Manual acceptance script (Step 1 check):**
  1. `drive --seed 1 --direction cw --route outer --record runs/` → stationary car, nose along road, panel v = 0.
  2. Without panel, heading identifiable in ≥ 4 poses at 64 px; H shows matching hi-res pose.
  3. Accelerate: smooth motion ≈ 1 px/tick at top speed, 10 m/s after ≈ 2.5 s; release holds speed; brake → 0, never reverses.
  4. 10 m/s into the Fork with full steer → runs wide → CRASH banner → auto-reset.
  5. Clean cw lap via outer branch → laps 1, last branch outer.
  6. Relaunch `drive --seed 2 --direction ccw --route inner --record runs/`, clean ccw lap via inner branch → laps 1, last branch inner.
  7. Turn around and drive against the episode Direction > 5 m → wrong_way on, no crash.
  8. Steer into the Island tip at the Fork → crash.
  9. Quit; `replay runs/drive_1.npz` prints PASS; `--show` reproduces the drive.
