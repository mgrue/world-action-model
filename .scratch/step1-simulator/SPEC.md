# Step 1 spec: `toycar` simulator

Buildable spec for PLAN.md Step 1. It covers a deterministic 2D overhead car simulator, a 64×64 RGB renderer and a keyboard driving mode. It also fixes the **interface contract** that Steps 2–4 (scripted expert, dataset, VAE) build on.

Sources: every decision here was made in a ticket of `MAP.md`, and each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root. Capitalised terms (Track, Route, Station, …) are defined there. Section 11 lists clarifications made while compiling that weren't asked explicitly.

---

## 1. Environment and project setup

([Python and package compatibility](issues/01-python-package-compatibility.md))

- **Python 3.14**: `requires-python = ">=3.14,<3.15"`, and `.python-version` = `3.14`. Managed with `uv`.
- **Runtime dependencies:** `numpy>=2.5,<2.6`, and `pygame-ce>=2.5.8,<2.6` (SDL2). **Never** install upstream `pygame`, because both provide `import pygame`.
- **Dev dependencies:** `pytest`.
- Torch is **not** a Step 1 dependency.

Layout:

```
pyproject.toml
src/toycar/
  __init__.py      re-exports the public API below; __version__
  track.py         TrackParams, Route, Track, Projection, Branch
  physics.py       constants, step_state
  state.py         State, StartSpec, Info, StepResult   (may live in env.py; public names matter, not files)
  render.py        RenderParams, render
  env.py           CarEnv
  drive.py         keyboard mode   (python -m toycar.drive)
  replay.py        replay tool     (python -m toycar.replay)
tests/
  golden/physics_trajectory.npz
  test_physics.py  test_track.py  test_render.py  test_env.py  test_replay.py
```

---

## 2. World and coordinates

([Track geometry, car size and look](issues/02-track-car-render-look.md))

- **Track units are metres.** The world is a fixed **40 m × 40 m** square, and that square is exactly the camera frame.
- **Origin** is at the top-left. **x** points right and **y** points **down**, the same as image coordinates.
- **Heading θ:** 0 points along +x. Positive θ rotates **clockwise on screen**, so +π/2 points along +y (down).
  - Forward unit vector: `f = (cos θ, sin θ)`.
  - Right-hand unit vector: `r = (−sin θ, cos θ)`.
- **Clockwise (cw)** on screen means travelling left→right along the top straight, then down the right side. Station increases in that direction.
- **Curvature sign:** positive curvature means turning clockwise (right). A path's heading changes by `dθ/ds = κ`.

---

## 3. Track

([Track geometry](issues/02-track-car-render-look.md), [Env API](issues/05-env-api-contract.md))

### 3.1 Parameters

```python
@dataclass(frozen=True)
class TrackParams:
    center: tuple[float, float] = (20.0, 20.0)
    straight_m: float = 18.0          # length of top and bottom straights
    end_radius_m: float = 7.5         # radius of left and right end arcs
    fork_length_m: float = 18.0       # chord length of the fork (= straight_m: fork spans the whole top straight)
    bulge_m: float = 6.0              # apex offset of each branch from the straight's line
    road_half_width_m: float = 2.5
    kerb_width_m: float = 0.3
```

Derived values for the defaults:

| Quantity | Formula | Value |
|---|---|---|
| α | `2·atan(2·bulge / fork_length)` | 1.176005 rad (67.380°) |
| ρ | `fork_length / (4·sin α)` | 4.875 m |
| branch length | `4·ρ·α` | 22.9321 m = `merge_station_m` |
| Route length | `branch + 2π·end_radius + straight` | **88.0560 m** |

Constraint for this version: `fork_length_m == straight_m`, so the Fork and Merge are the two ends of the top straight. Validate it.

### 3.2 Routes (built from analytic primitives)

Both Routes start at the **Fork** `F = (cx − straight/2, cy − end_radius)` = (11, 12.5) with heading 0, and run clockwise:

| # | Primitive | Outer route | Inner route |
|---|---|---|---|
| 1 | arc, radius ρ | turn −α | turn +α |
| 2 | arc, radius ρ | turn +2α | turn −2α |
| 3 | arc, radius ρ | turn −α | turn +α |
| 4 | arc, radius end_radius | turn +π (right end, centre (29, 20)) | same |
| 5 | line, length straight | heading π (bottom straight, y = 27.5) | same |
| 6 | arc, radius end_radius | turn +π (left end, centre (11, 20)) | same |

How the table reads:
- Primitives 1–3 form a **Branch**. It ends at the **Merge** M = (29, 12.5) with heading 0.
- The outer branch bulges up (−y), to an apex at y = 6.5. The inner branch bulges down (+y), to an apex at y = 18.5.
- A turn of angle φ along an arc of radius R has length `R·|φ|` and curvature `sign(φ)/R`.
- The Route closes exactly at F with heading 0 mod 2π. Assert this to within 1e-9 when building.
- **Station** `s ∈ [0, length)` is arc length from F along the primitives. The two Routes have identical length and identical Station parametrisation for `s ≥ merge_station_m`.

### 3.3 API

```python
Branch = Literal["outer", "inner"]

@dataclass(frozen=True)
class Projection:
    station_m: float     # s of the closest centerline point on this Route
    lateral_m: float     # signed offset along the Route's right-hand normal (+ = right of the cw tangent)
    distance_m: float    # |lateral_m|  (distance to the closest point)

class Route:
    name: Branch
    length_m: float
    def pose_at(self, s: float) -> tuple[float, float, float]   # x, y, cw tangent heading; s taken mod length
    def curvature_at(self, s: float) -> float                   # 1/m, + = clockwise
    def project(self, x: float, y: float) -> Projection         # exact: closest point over all primitives

class Track:
    params: TrackParams
    outer: Route
    inner: Route
    length_m: float                 # route length (≈ 88.056)
    fork_station_m: float           # 0.0
    merge_station_m: float          # branch length (≈ 22.932)
    def __init__(self, params: TrackParams = TrackParams()) -> None   # builds and validates
    @classmethod
    def default(cls) -> "Track"
    def route(self, name: Branch) -> Route
    def locate(self, x: float, y: float) -> tuple[float, Branch | None]
    def distance_to_centerline(self, x: float, y: float) -> float   # min over both Routes
    def is_crash(self, state: "State") -> bool
```

Behaviour:
- **`project`** computes the exact closest point on each line or arc primitive and takes the minimum.
  - Closest point on an arc: clamp the angle of `(p − centre)` to the arc's angular range, then also compare both endpoints.
  - Ties go to the lower Station.
- **`locate(x, y)`** projects onto both Routes and picks the one with the smaller `distance_m` (ties go to `"outer"`). It returns that Station, and returns the branch name if `fork_station_m ≤ s < merge_station_m`, otherwise `None`.
- **`is_crash(state)`** is True if **any of the 4 car corners** is more than `road_half_width_m + kerb_width_m` (2.8 m) from the nearest point on *either* Route centerline. The corners are `(x, y) ± (CAR_LENGTH/2)·f ± (CAR_WIDTH/2)·r`.

Validation in `__init__` raises `ValueError` if any of these fails:
- the whole road including the kerb lies within `[0, 40]²`
- the island is open: `2·bulge − 2·(road_half_width + kerb_width) > 0`
- the inner branch's kerb is clear of the bottom straight's kerb
- `fork_length == straight`
- `ρ > R_MIN` (the minimum turning radius in §4)

---

## 4. Physics

([Physics model](issues/03-physics-model.md))

This is a kinematic bicycle model:
- steering is instant and throttle is an acceleration command
- there is no tyre slip, no drag and no reverse
- `(x, y)` is the kinematic reference point and also the geometric centre of the car rectangle (§11)

Constants (in `physics.py`):

```python
DT = 1 / 15                 # s, 15 Hz; one Tick = one Action, one State update, one Frame
WHEELBASE = 2.7             # m
DELTA_MAX = math.radians(35)        # R_MIN = WHEELBASE / tan(DELTA_MAX) ≈ 3.856 m
A_LAT_MAX = 8.0             # m/s²
ACCEL = 4.0                 # m/s² at throttle +1
BRAKE = 8.0                 # m/s² at throttle −1
V_MAX = 10.0                # m/s
CAR_LENGTH, CAR_WIDTH = 4.4, 2.5    # m
```

`State` and the Action:

```python
@dataclass(frozen=True)
class State:
    x: float; y: float; theta: float; v: float      # float64 (Python float); theta in (−π, π]
    def features(self) -> np.ndarray:               # float32 shape (5,)
        return np.array([x/40, y/40, cos θ, sin θ, v/10], dtype=np.float32)
```

The Action is any length-2 sequence `[steer, throttle]`, converted to Python floats.

**`step_state(state: State, action) -> State`** is pure and uses **Python `float` and `math` only, never numpy**. The operations must run in exactly this order:

```
1. steer    = min(max(float(action[0]), -1.0), 1.0)
   throttle = min(max(float(action[1]), -1.0), 1.0)
2. a  = ACCEL * throttle if throttle >= 0 else BRAKE * throttle
3. v1 = min(max(v + a * DT, 0.0), V_MAX)
4. t  = math.tan(steer * DELTA_MAX)
   if v1 > 0:
       T = min(math.tan(DELTA_MAX), A_LAT_MAX * WHEELBASE / (v1 * v1))
       t = min(max(t, -T), T)
5. k  = t / WHEELBASE
6. th1 = theta + v1 * k * DT
   if abs(k) > 1e-9:
       x1 = x + (math.sin(th1) - math.sin(theta)) / k
       y1 = y - (math.cos(th1) - math.cos(theta)) / k
   else:
       x1 = x + v1 * DT * math.cos(theta)
       y1 = y + v1 * DT * math.sin(theta)
7. th1 = wrap(th1)   where wrap(a) = a - 2π·ceil((a - π) / (2π))  → (−π, π]
return State(x1, y1, th1, v1)
```

These numbers give curve speed limits of about 6.24 m/s on the branch arcs (`√(A_LAT_MAX·ρ)`) and about 7.75 m/s on the end arcs. Too much speed makes the car understeer wide.

---

## 5. Rendering

([Track geometry, car size and look](issues/02-track-car-render-look.md), [Env API](issues/05-env-api-contract.md))

```python
RGB = tuple[int, int, int]

@dataclass(frozen=True)
class RenderParams:
    grass: RGB = (46, 110, 52)
    road:  RGB = (70, 70, 76)
    kerb:  RGB = (200, 200, 200)
    body:  RGB = (220, 40, 40)
    nose:  RGB = (255, 235, 60)
    supersample: int = 4

def render(state: State, track: Track, params: RenderParams = RenderParams(),
           resolution: int = 64) -> np.ndarray     # uint8, shape (resolution, resolution, 3), C-order
```

`render` is pure: the output depends only on its arguments.

1. **Sample grid.** Let `n = resolution · supersample`. Sub-pixel `(i, j)` (row i, column j) samples the world point `((j + 0.5)·40/n, (i + 0.5)·40/n)`.
2. **Background**, per sub-pixel, using `d = track.distance_to_centerline(point)`:
   - `d < road_half_width` → road
   - `d < road_half_width + kerb_width` → kerb
   - otherwise → grass

   The island and infield are plain grass. There are no other markings (no start line, divider or arrows).
3. **Car** is drawn over the background. A sub-pixel is car if `|fwd| < CAR_LENGTH/2` and `|lat| < CAR_WIDTH/2`, where `fwd = (p − c)·f` and `lat = (p − c)·r`. It's **nose** colour if additionally `fwd > CAR_LENGTH/2 − 0.35·CAR_LENGTH`, otherwise body colour.
4. **Downsample.** Average each `supersample × supersample` block in float64, then convert with `np.round` (half to even) and cast to uint8.
5. **Caching** is a performance requirement:
   - Cache the background (sub-pixel colour array) per `(track.params, params, resolution)`.
   - Per frame, recompute only the sub-pixels inside the car's bounding box.
   - Target: under 2 ms per 64×64 frame on an M-series Mac, after the cache is warm.

Frames contain no speed cue or trail. Speed reaches models only through `State.features()`.

---

## 6. Environment

([Env API and interface contract](issues/05-env-api-contract.md), [What happens when the car leaves the road](issues/04-off-track-semantics.md))

```python
Direction = Literal["cw", "ccw"]
Termination = Literal["crash", "max_steps"] | None

@dataclass(frozen=True)
class StartSpec:                       # every field optional; None = sample it
    direction: Direction | None = None
    route: Branch | None = None
    station_m: float | None = None
    lateral_m: float | None = None
    heading_offset: float | None = None   # radians, relative to the Route tangent in the episode direction
    speed: float | None = None

@dataclass(frozen=True)
class Info:
    station_m: float
    progress_m: float
    laps: int
    direction: Direction
    wrong_way: bool
    branch: Branch | None
    last_branch: Branch | None
    steps: int
    termination: Termination

@dataclass(frozen=True)
class StepResult:
    state: State
    terminated: bool
    truncated: bool
    info: Info

class CarEnv:
    def __init__(self, track: Track | None = None, render_params: RenderParams | None = None,
                 max_steps: int = 900, resolution: int = 64) -> None
    def reset(self, seed: int, start: StartSpec | None = None,
              render_params: RenderParams | None = None) -> tuple[State, Info]
    def step(self, action: Sequence[float]) -> StepResult
    def render(self) -> np.ndarray                     # render(self.state, track, current params, resolution)
    @property
    def state(self) -> State
    @property
    def track(self) -> Track
```

There is no gymnasium dependency and no reward. `step` doesn't return frames: call `render()` when you need one.

### 6.1 `reset(seed, start, render_params)`

`seed` is required. Reset does the following:

1. Create `rng = np.random.default_rng(seed)`.
2. Draw **all six** values in this fixed order, **even if `start` overrides them**, so overriding one field never changes the others:
   1. `direction = ["cw","ccw"][rng.integers(2)]`
   2. `route = ["outer","inner"][rng.integers(2)]`
   3. `station = rng.uniform(0, track.length_m)`
   4. `lateral = rng.uniform(-1.0, 1.0)`
   5. `heading_offset = rng.uniform(-π/18, π/18)` (±10°)
   6. `speed = rng.uniform(0.0, 6.0)`
3. Apply the `StartSpec` overrides.
4. Build the pose:
   - Take `(px, py, tθ) = route.pose_at(station)` and its right normal `r(tθ)`.
   - Position = `(px, py) + lateral · r`.
   - θ = `wrap(tθ + (π if direction == "ccw" else 0) + heading_offset)`.
   - v = speed.
5. If `track.is_crash(pose)`: redraw all six values from the same `rng` (overrides re-applied) until it isn't. If 1000 attempts fail (only possible with a bad StartSpec), raise `ValueError`.
6. Reset the bookkeeping: `steps = 0`, `progress_m = 0`, `max_progress_m = 0`, `laps = 0`, `wrong_way = False`, `branch` from `locate`, and `last_branch = None`. Set `termination = None`.
7. If `render_params` is given, it replaces the env's params for this episode. Otherwise the constructor's params (or the defaults) apply.

### 6.2 `step(action)`

1. If the episode has already ended (terminated or truncated), raise `RuntimeError("episode ended; call reset()")`.
2. `state = step_state(state, action)` and `steps += 1`.
3. Update progress:
   - `(s_new, branch_new) = track.locate(x, y)`
   - `Δ = s_new − s_old`. If `Δ > L/2`, subtract L; if `Δ < −L/2`, add L, where L = `track.length_m`.
   - `progress_m += Δ if direction == "cw" else −Δ`
   - `max_progress_m = max(max_progress_m, progress_m)`
   - `laps = max(0, floor(progress_m / L))`
   - `wrong_way = progress_m < max_progress_m − 5.0`
4. Update the branch:
   - If the previous `branch` was not None and `branch_new` is None, the car has left the fork section.
   - If it left at the far end for its travel sense (`Δ > 0` exiting past the Merge, `Δ < 0` exiting past the Fork), set `last_branch = previous branch`.
   - Otherwise (it backed out the way it came), leave `last_branch` unchanged.
5. Check for a crash: if `track.is_crash(state)`, set `terminated = True` and `termination = "crash"`.
6. Check truncation: else if `steps >= max_steps`, set `truncated = True` and `termination = "max_steps"`.
7. Return `StepResult`. A crashing tick still returns its State, and `render()` then shows the car touching grass.

Standing still on the road and driving the wrong way are **not** crashes. A stopped car's episode is ended only by `max_steps`.

---

## 7. Keyboard driving mode

([Keyboard driving mode](issues/06-keyboard-driving-mode.md))

`uv run python -m toycar.drive [--seed N] [--direction cw|ccw] [--route outer|inner] [--speed V] [--record DIR]`

**Seed and start.**
- Without `--seed`, the seed is `int(time.time())`, printed to the terminal and shown in the panel.
- `--direction`, `--route` and `--speed` fill `StartSpec`. **The default start speed is 0.**
- `max_steps = 10**9`.

**Loop.**
- Fixed 15 Hz using `pygame.time.Clock().tick(15)`.
- Each iteration: pump events (`pygame.event.get()`), read held keys with `pygame.key.get_pressed()`, build the action, call `env.step` once, then `env.render()`. One loop iteration is one Tick.

**Input** (lives only in `drive.py`; physics stays instant-steer; recorded actions are the values passed to `step`):

| Key | Effect |
|---|---|
| ↑ / W | throttle = +1 |
| ↓ / S | throttle = −1 (brake; never reverse) |
| neither | throttle = 0 (cruise; no drag) |
| ← / A | steer ramps toward −1 at 4 /s (0.25 s to full) |
| → / D | steer ramps toward +1 at 4 /s |
| no steer key (or both) | steer ramps toward 0 at 6 /s |
| H | toggle hi-res main view: `render(state, track, params, resolution=512)` |
| Tab | toggle the debug panel |
| P | pause |
| R | reset with seed + 1 |
| Shift+R | restart the same seed |
| Esc / Q | quit |

**Display.**
- The main view is the true 64×64 frame upscaled ×8 with nearest-neighbour scaling, giving 512×512.
- Build it with `pygame.surfarray.make_surface(frame.swapaxes(0, 1))` and `pygame.transform.scale`. Don't use `smoothscale`.
- The debug panel (about 220 px wide) sits to the **right** of the frame, never over it. It shows x, y, θ in degrees, v, steer, throttle, station, progress, laps, direction, branch, last_branch, wrong_way, steps and seed.
- The window is about 732×512. On Retina displays macOS may upscale the window 2× with slight softening, which is accepted.

**Crash.** Freeze the last frame with a red **"CRASH"** banner for 1.5 s, then auto-reset with seed + 1.

**Recording.** With `--record DIR`, when each episode ends (crash, R, Shift+R or quit) the tool writes `DIR/drive_<seed>.npz` (`np.savez_compressed`) containing:
- `meta`: a JSON string with `seed`, `start` (the StartSpec as a dict), `render_params`, `track_params`, `toycar_version` and `termination` (`"crash"`, `"reset"` or `"quit"`)
- `actions`: float64 `(N, 2)`
- `states`: float64 `(N+1, 4)` as `(x, y, θ, v)`, including the reset State
- `frame_sha256`: `(N+1,)` hex strings, the SHA-256 of each 64×64 frame's bytes

**Replay.** `uv run python -m toycar.replay FILE [--show] [--export-png DIR]`
- **Default:** rebuild `Track(track_params)` and `CarEnv`, call `reset(seed, start, render_params)`, apply the actions, and compare every State (exact float equality) and frame hash. Print `PASS`, or `FAIL at tick k` plus the differing field, and exit non-zero on failure.
- `--show` plays the drive back in the window at 15 Hz.
- `--export-png DIR` writes `DIR/frame_00000.png`, … at 64×64. PNG writing uses `pygame.image.save` to avoid an extra dependency.

---

## 8. Determinism

([Physics model](issues/03-physics-model.md))

- **The guarantee:** for the same seed, StartSpec, RenderParams, TrackParams and Action sequence, the results are **bit-identical States and Frames on the same machine with pinned Python and numpy versions**.
- Nothing is guaranteed across platforms. The Step 3 dataset is recorded once and stored, so nothing needs to replay elsewhere.
- **Physics:** Python `float` and `math` only.
- **Randomness:** `np.random.default_rng(seed)` only, with the fixed draw order in §6.1. No global RNG and no time-dependent behaviour in `toycar` outside `drive.py`'s default seed.
- **Rendering:** pure, with float64 accumulation and `np.round`.

---

## 9. Automated tests (pytest)

| Test | Asserts |
|---|---|
| `test_physics_golden` | Start from `State(20.0, 27.5, math.pi, 5.0)` and apply the 300 actions stored in `tests/golden/physics_trajectory.npz` with pure `step_state` (no track, so no crash). All 301 States match the stored ones within **1e-9**. To generate the golden file: actions = `np.random.default_rng(0).uniform(-1, 1, (300, 2))`, run once, commit. |
| `test_physics_properties` | v stays in [0, 10]; throttle 0 keeps v unchanged; steer 0 gives a straight line; at v = 10 with full steer the turning radius is `v²/A_LAT_MAX` = 12.5 m (clamped); at v = 2 with full steer it's R_MIN; braking never makes v negative; θ stays in (−π, π]. |
| `test_track_geometry` | both Routes close within 1e-9; their lengths are equal (≈ 88.056); `pose_at` and `project` round-trip (`project(pose_at(s))` gives `(s, 0, 0)`); `curvature_at` equals ±1/ρ, +1/end_radius or 0 on the right primitives; outer apex at y = 6.5 and inner apex at y = 18.5; `TrackParams` validation rejects an oversized bulge. |
| `test_crash_predicate` | a centred car on any Station isn't a crash; a car with its centre at lateral 1.6 m and θ along the tangent on a straight is a crash (corner at 2.85 m); a car centred at the Island centre (20, 12.5) is a crash. |
| `test_env_determinism` | two `CarEnv`s with the same seed and 500 identical random actions give identical State sequences and identical SHA-256 hashes of every `render()`. |
| `test_start_pose_validity` | for seeds 0..9999, `reset` never returns a crash pose, speed is in [0, 6], and the overrides are honoured; overriding `speed` doesn't change the sampled station. |
| `test_progress_and_laps` | drive the outer Route with an ideal controller (or teleport States along `pose_at`) cw for 2.5 lengths: laps = 2, wrong_way stays False, last_branch = "outer". The same ccw via inner gives last_branch = "inner". Reversing for 6 m sets wrong_way. |
| `test_episode_end` | `max_steps = 5` gives truncated with "max_steps" at step 5; stepping after the end raises `RuntimeError`; a crash gives terminated with "crash". |
| `test_render` | shape and dtype; the 64-px pixel containing world point (20, 12.5) (Island centre) is grass; a car pose changes only pixels near the car; a 0.25 m move changes at least one pixel; for a car at θ = 0 on a straight, at least 6 pixels nearer the nose colour and 8 nearer the body colour at 64 px. |
| `test_replay_roundtrip` | record a synthetic episode in the drive `.npz` format, then `replay` (without `--show`) returns PASS. |

---

## 10. Manual acceptance (PLAN.md Step 1 check)

([Keyboard driving mode](issues/06-keyboard-driving-mode.md))

"A human can drive the car with the keyboard and the rendering looks right" passes when every line below passes:

1. `drive --seed 1 --direction cw --route outer --record runs/` shows a stationary car with its nose along the road, and the panel shows v = 0.
2. With the panel hidden, the tester can tell the heading in at least 4 poses at 64 px. **H** shows the same pose in hi-res.
3. **Accelerate:** the motion is smooth, about 1 px per tick at top speed, and v reaches 10 m/s after about 2.5 s. Releasing the keys holds speed. Braking brings v to 0 and the car never reverses.
4. At 10 m/s into the Fork with full steer, the car runs wide, the CRASH banner shows, and it auto-resets.
5. A clean cw lap via the outer branch gives laps 1 and last branch outer.
6. `drive --seed 2 --direction ccw --route inner --record runs/` followed by a clean ccw lap via the inner branch gives laps 1 and last branch inner.
7. Turning around and driving against the episode Direction for more than 5 m turns wrong_way on, with no crash.
8. Steering into the Island tip at the Fork is a crash.
9. Quit, then `replay runs/drive_1.npz` prints PASS, and `--show` reproduces the drive.

---

## 11. Clarifications made while compiling

These fill gaps between ticket decisions. They don't change any decision. If one looks wrong, reopen the relevant ticket.

1. **Reference point.** `(x, y)` is both the bicycle model's kinematic reference point and the car rectangle's centre. The equations are the rear-axle form with no side-slip angle. That's a deliberate simplification: the spec's equations are authoritative.
2. **Exact lengths.** The Route length is computed (≈ 88.056 m, not 88.0) and `merge_station_m` ≈ 22.932. Laps and station ranges use the computed length.
3. **Tie-breaking.** In `locate`, ties go to the outer Route. In `project`, ties go to the lower Station.
4. **Station unwrapping** uses the ±L/2 rule. `last_branch` updates only on exiting the fork section at the far end for the travel sense.
5. **RNG draw order** is fixed, and every value is drawn even when overridden.
6. **Reset retries** stop after 1000 attempts with `ValueError`.
7. **After truncation**, `step()` raises, the same as after a crash.
8. **Recording** includes the reasons `"reset"` and `"quit"` alongside `"crash"`.
9. **Performance target:** under 2 ms per 64 px frame with the background cache.
10. **Crash test numbers** (§9) follow from geometry. On a straight with θ along the tangent, the corner offset is lateral + 1.25 m, so lateral 1.6 m puts it at 2.85 m, more than 2.8.

## 12. Out of scope for Step 1

- Scripted expert (Step 2), dataset recording format and visual-randomisation ranges (Step 3), VAE (Step 4), frame history and speed observability (Steps 3/6).
- Step 1 only provides the interface those steps use: Track queries, `step_state`, `CarEnv`, `RenderParams` and `render`.
