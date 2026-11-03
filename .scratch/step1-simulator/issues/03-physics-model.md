# Physics model and action/state definitions

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 02-track-car-render-look.md
Parent: ../MAP.md

## Question

Which vehicle model (kinematic bicycle vs unicycle, with or without steering/throttle inertia), tick rate, units, action ranges and semantics (steer, throttle — bounded `[-1,1]`? braking/reverse?), max speed and turning radius relative to the chosen track, and exact state vector (position, heading, speed — anything else?) should the simulator use? How is heading represented in state (angle vs sin/cos)? Integration scheme and dtype rules that guarantee determinism.

Constraint from the track ticket: the tightest curve is the 4.88 m branch arc, so the minimum turning radius must sit comfortably below it; track units are metres, y down, heading clockwise-positive.

Check: the spec states the step equations explicitly enough that two independent implementations would match.

## Comments

- Q1 vehicle model → **kinematic bicycle, instant steering** (action sets front-wheel angle directly, no steering lag), throttle is an acceleration command; no tyre slip/drift. Wheelbase 2.7 m (pending max-steer question).
- Q2 tick rate → **15 Hz** (dt = 1/15 s): one tick = one physics step, one frame, one action. 16-tick chunk ≈ 1.07 s.
- Q3 grip → **lateral-acceleration limit with clamping**: |tan δ| ≤ min(tan δ_max, a_lat_max·L/v²); commanded angle beyond it is clamped (car understeers onto a wider arc). Makes throttle meaningful (brake before Fork / end arcs) and gives a natural way to run wide off the road.
- Q4 numbers → action = [steer, throttle] ∈ [-1,1]² (clipped). δ_cmd = steer·δ_max, δ_max = 35°, wheelbase L = 2.7 m (R_min ≈ 3.86 m). a_lat_max = 8 m/s². throttle>0 → a = throttle·4 m/s², throttle<0 → a = throttle·8 m/s². Speed clamped to [0, 10] m/s, no reverse, no drag.
- Q5 state → canonical **State = (x, y, θ, v)** float64; θ wrapped to (−π, π], 0 = +x, clockwise positive; nothing else (no steering angle, previous action or lap counter — progress/laps are info, Env API ticket). Model-facing **State features = (x/40, y/40, cos θ, sin θ, v/10)**.
- Q6 step equations → per tick (dt = 1/15): clip action; a = 4·throttle (≥0) or 8·throttle (<0); v' = clip(v + a·dt, 0, 10) (speed first, semi-implicit); t = tan(steer·δ_max), if v'>0 clip t to ±min(tan δ_max, a_lat·L/v'²); κ = t/L; θ' = θ + v'·κ·dt; **exact arc**: x' = x + (sin θ' − sin θ)/κ, y' = y − (cos θ' − cos θ)/κ, straight-line fallback when |κ| ≤ 1e-9; wrap θ' to (−π, π]. No substeps.
- Q7 determinism → bit-identical States and Frames for same seed + actions **on the same machine with pinned Python/numpy**; physics uses Python `float` + `math` (no numpy) ; renderer is a pure function of (State, render params) → uint8. Tests: exact hash equality within a machine; committed golden trajectory (seed + 300 actions → States) compared at tolerance 1e-9 across machines. No cross-platform bit-exactness guarantee (dataset is recorded once and stored).

**Resolution (2026-11-03):**
- **Model:** kinematic bicycle, instant steering, acceleration throttle, no slip, no drag, no reverse.
- **Timing:** 15 Hz, dt = 1/15 s; one tick = one Action, one State update, one Frame.
- **Action:** [steer, throttle] ∈ [-1,1]², clipped.
- **Constants:** L = 2.7 m, δ_max = 35° (R_min ≈ 3.86 m), a_lat_max = 8 m/s², accel 4 m/s², brake 8 m/s², v ∈ [0, 10] m/s.
- **State:** (x, y, θ, v) float64, θ ∈ (−π, π], 0 = +x, clockwise positive. State features for models: (x/40, y/40, cos θ, sin θ, v/10).
- **Step equations** (in order): clip → a → v' = clip(v + a·dt, 0, 10) → t = tan(steer·δ_max), clipped to ±min(tan δ_max, a_lat·L/v'²) when v' > 0 → κ = t/L → θ' = θ + v'κdt → exact arc x' = x + (sin θ' − sin θ)/κ, y' = y − (cos θ' − cos θ)/κ (straight fallback |κ| ≤ 1e-9) → wrap θ'.
- **Determinism:** as Q7 above.
- Glossary: State, State features, Action, Tick added to CONTEXT.md.
