# Step 2 spec: `toycar.expert` scripted expert

Buildable spec for PLAN.md Step 2: a scripted expert that drives the Step 1 `toycar` simulator under three Instructions. It also fixes the **interface contract** that Step 3 (dataset recording) and Step 9 (closed-loop evaluation) build on: the Instruction labels, the expert API, demonstration noise, and a policy-agnostic episode scorer.

Sources: every decision here was made in a ticket of `MAP.md`, and each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root (Track, Route, Station, Passage, Instruction, Planned branch, …). The Step 1 contract is `.scratch/step1-simulator/SPEC.md`; this spec only adds to it. The reference implementation is the throwaway prototype `prototypes/expert_prototype.py`; where this spec and the prototype differ, this spec wins. Section 11 lists clarifications made while compiling that weren't asked explicitly.

PLAN.md check: "the expert completes laps reliably for every behavior" (§9, §10).

---

## 1. Scope and layout

- New module **`src/toycar/expert.py`** (plus `__main__`-style CLI in the same module: `python -m toycar.expert`). No new dependencies: Python `float`/`math` for control, `numpy` for RNG only.
- `toycar/__init__.py` re-exports `Instruction`, `INSTRUCTIONS`, `ExpertParams`, `ExpertAction`, `Expert`, `EpisodeScore`, `score_episode`, `summarize`.
- New tests: `tests/test_expert.py`, `tests/test_score.py`.
- Step 1 dependencies: `Track`, `Route.pose_at/curvature_at/project`, `step_state`, `wrap`, physics constants, `CarEnv`, `Info` (including **`Info.passages`** and the midpoint-based **`last_branch`**, both added to Step 1 during this map; see Step 1 SPEC §6.2).

---

## 2. Instructions

([Instruction set and branch choice](issues/02-instruction-set.md))

```python
Instruction = Literal["outer", "inner", "any"]
INSTRUCTIONS: tuple[Instruction, ...] = ("outer", "inner", "any")   # index = integer id; order frozen, append only
```

| Instruction | id | Branch rule |
|---|---|---|
| `outer` | 0 | take the outer branch on every Passage |
| `inner` | 1 | take the inner branch on every Passage |
| `any` | 2 | independent coin per Passage: outer with probability `p_outer` (default 0.5) |

- **Direction is not part of the Instruction.** It is the episode Direction from `CarEnv.reset`, and both Directions are valid under every Instruction.
- The Instruction is **fixed for the whole episode**.
- **Recording rule (for Step 3 and `eval`):** under `outer`/`inner` the caller resets with `StartSpec(route=instruction)`. Outside the fork section the two Routes coincide, so this changes nothing else. Under `any` the start route stays random.
- The classifier-free-guidance "null" label is **model-side only** (embedding index `len(INSTRUCTIONS)` = 3). The expert never receives it and the data never contains it.
- Free-text phrasings are deferred to PLAN Step 5.

---

## 3. Parameters

([Expert variation knobs](issues/04-expert-variation.md))

```python
@dataclass(frozen=True)
class ExpertParams:
    a_lat_plan: float = 5.0      # m/s², lateral accel used to plan curve speeds
    v_cap: float = 10.0          # m/s, speed-profile cap
    burst_peak: float = 0.4      # max steer amplitude of a noise burst; 0 = clean
    burst_gap_s: float = 3.0     # s, mean of the exponential gap between bursts
    p_outer: float = 0.5         # P(outer) per coin under "any"
```

| Knob | Safe range (0 crashes in 6000 episodes) |
|---|---|
| `a_lat_plan` | [4.0, 5.0]. 5.5 is safe only with `burst_gap_s ≥ 3`; 6.0 crashes |
| `v_cap` | [6, 10] |
| `burst_peak` | [0, 0.4]. 0.5 → 3–4 crashes, 0.6 → 64 |
| `burst_gap_s` | ≥ 1.0 |
| `p_outer` | [0, 1] |

`__init__` of `Expert` raises `ValueError` outside `a_lat_plan ∈ (0, 8)`, `v_cap ∈ (0, 10]`, `burst_peak ∈ [0, 1]`, `burst_gap_s > 0`, `p_outer ∈ [0, 1]`. It does **not** enforce the safe ranges; `eval` is how a setting is checked.

**Hidden-variable caveat (for Step 3).** `a_lat_plan` and `v_cap` are per-episode style the model cannot see: the same State gets different labels under different values. Keep them **fixed** unless varying them is the experiment.

Constants (module level, not parameters):

```python
WS = 0.5                 # 1/m, steering bandwidth: ke = WS², kθ = 2·WS (critically damped)
A_ACC_PLAN = 4.0         # m/s², profile forward pass (= ACCEL)
A_BRK_PLAN = 6.0         # m/s², profile backward pass (2 m/s² below BRAKE)
PROFILE_DS = 0.05        # m, profile grid spacing
SPEED_LOOKAHEAD_M = 0.5  # m, profile look-ahead beyond one tick
ROLLOUT_S = 3.0          # s, start-branch feasibility rollout
WRONG_BRANCH_M = 2.5     # m, off-Route threshold for the rollout and the safety rule
GRACE_S = 2.0            # s, no noise bursts after reset
```

---

## 4. Control law (clean)

([Controller research](issues/01-controller-research.md), [Controller prototype](issues/03-controller-prototype.md); details in `research/01-controller-and-demo-noise.md`)

`Expert.control(state, branch, direction) -> (steer, throttle)` is **pure**: it depends only on its arguments, the Track and the params, and it never touches RNG or episode state. Python `float` and `math` only. Let `route = track.route(branch)` and `d = +1` for cw, `−1` for ccw. In this order:

```
1. pr  = route.project(x, y);  s = pr.station_m
   e   = d · pr.lateral_m                                   # + = right of the travel direction
   (_, _, th) = route.pose_at(s)
   θe  = wrap(θ − th − (π if d < 0 else 0))
2. vref = profile(branch, d)(s + d·(v·DT + SPEED_LOOKAHEAD_M))      # §4.1
   a    = (vref − v) / DT
   throttle = min(1, a/ACCEL) if a ≥ 0 else max(−1, a/BRAKE)
3. v1 = min(max(v + (ACCEL·throttle if throttle ≥ 0 else BRAKE·throttle)·DT, 0), V_MAX)   # = step_state's v1
4. dist = max(v1·DT, 1e-3)
   κ̄   = wrap(pose_at(s + d·dist).θ − pose_at(s).θ) / dist  # travel-signed mean curvature over the next tick
5. sinc = sin θe / θe if |θe| > 1e-6 else 1
   k    = κ̄·cos θe / max(1 − κ̄·e, 0.2)  −  2·WS·θe  −  WS²·e·sinc
6. T = tan(DELTA_MAX) if v1 ≤ 0 else min(tan(DELTA_MAX), A_LAT_MAX·WHEELBASE / v1²)
   k = clamp(k, −T/WHEELBASE, T/WHEELBASE)
7. steer = clamp(atan(WHEELBASE·k) / DELTA_MAX, −1, 1)
```

Why it works: the State reference point turns with exactly the commanded curvature (`k = tan δ / L`) and steering is instant, so the feed-forward term is exact. Feedback only removes start offsets and noise. Using the *mean* curvature over the next tick handles the S-bend sign flips at the branch arcs. Pure pursuit (named in PLAN.md) was rejected: it held 0.13–0.55 m RMS error, against about 1 mm for this law.

### 4.1 Speed profile

One profile per `(branch, d, a_lat_plan, v_cap)`, built once from the expert's Track and cached:

```
n   = ceil(route.length_m / PROFILE_DS)
s_i = (d · i · PROFILE_DS) mod length,  i = 0..n−1        # grid in travel order
v_i = min(v_cap, sqrt(a_lat_plan / max(|curvature_at(s_i)|, 1e-9)))
repeat 2 times:                                            # closed loop: wrap-around consistent
    forward  i = 1..2n−1    (j = i mod n, p = (i−1) mod n):   v_j = min(v_j, sqrt(v_p² + 2·A_ACC_PLAN·ds))
    backward i = 2n−2..0    (j = i mod n, q = (i+1) mod n):   v_j = min(v_j, sqrt(v_q² + 2·A_BRK_PLAN·ds))
lookup(s) = v[round(((d·s) mod length) / ds) mod n]
```

At the defaults: 4.94 m/s on branch arcs, 6.12 m/s on end arcs, briefly 10 m/s on the bottom straight; about 4.1 laps per 900-tick episode.

---

## 5. Expert

([Instruction set](issues/02-instruction-set.md), [Variation knobs](issues/04-expert-variation.md), [API](issues/05-expert-api-and-metrics.md))

```python
@dataclass(frozen=True)
class ExpertAction:
    clean: tuple[float, float]       # expert label: control(state, planned_branch, direction)
    executed: tuple[float, float]    # clean + noise, clipped; pass this to env.step
    perturbed: bool                  # True iff a burst is active on this tick

class Expert:
    def __init__(self, track: Track, instruction: Instruction, params: ExpertParams = ExpertParams()) -> None
    def reset(self, seed: int, state: State, info: Info) -> None     # call right after env.reset, same seed
    def act(self, state: State, info: Info) -> ExpertAction          # exactly once per tick
    @property
    def planned_branch(self) -> Branch
    @property
    def safety_switches(self) -> int                                  # times the §5.3 safety rule fired this episode
    def control(self, state: State, branch: Branch, direction: Direction) -> tuple[float, float]   # §4
```

The expert reads **State and Info only**, never Frames. Caller loop:

```python
state, info = env.reset(seed, StartSpec(direction=..., route=None if instr == "any" else instr))
expert.reset(seed, state, info)
while True:
    a = expert.act(state, info)
    r = env.step(a.executed)
    state, info = r.state, r.info
    if r.terminated or r.truncated:
        break
```

Calling `act` before `reset` raises `RuntimeError`.

### 5.1 Random streams

- **Coin:** `coin_rng = np.random.default_rng([seed, 1])`. Each draw is `"outer" if coin_rng.random() < p_outer else "inner"`, and is taken **only under `any`**.
- **Noise:** `noise_rng = np.random.default_rng([seed, 2])`. Nothing is drawn when `burst_peak == 0`.
- Both are disjoint from the env's `default_rng(seed)`. Changing noise knobs, or turning noise off, never changes the `any` branch sequence.
- An episode is reproducible from (seed, StartSpec, Instruction, ExpertParams).

### 5.2 Planned branch

`preferred()` is the Instruction under `outer`/`inner`, or one coin draw under `any`.

**At `reset`:**
1. `pref = preferred()` (exactly one coin draw under `any`), `other` = the other branch.
2. **Feasibility rollout** `feasible(b)`: false if the start state is more than `WRONG_BRANCH_M` from `route(b)`. Otherwise iterate `st = step_state(st, control(st, b, direction))` for `round(ROLLOUT_S / DT)` = 45 ticks; false if any tick Crashes (`track.is_crash`) or is more than `WRONG_BRANCH_M` from `route(b)`; else true.
3. `planned = other if (not feasible(pref) and feasible(other)) else pref`.
4. Remember `prev_branch = info.branch`; `safety_switches = 0`.

This replaces nearest-projection and heading-threshold rules, both of which crashed near the Fork and Merge, where the branches coincide in position and differ by up to 67° in heading.

**In `act`, step A, section exit:** if `prev_branch is not None and info.branch is None`, the car has left the fork section (either end): `planned = preferred()` (one coin draw under `any`). Then set `prev_branch = info.branch`.

The coin draw order is therefore: one at reset, then one per section exit. The expert follows `route(planned)` everywhere; outside the fork section both Routes coincide, so switching the plan there has no effect on the path.

### 5.3 Safety rule

**In `act`, step B:** if `info.branch is not None and info.branch != planned and route(planned).project(x, y).distance_m > WRONG_BRANCH_M`, then `planned = info.branch` and `safety_switches += 1`. The car is clearly in the other branch and can't cross the Island, so it finishes that one.

It never fires for the expert in clean or noisy sweeps; it exists for robustness and Step 9 (for example a learned policy handing control to the expert).

### 5.4 Demonstration noise

**In `act`, step C:** `clean = control(state, planned, info.direction)`.

**Step D: steer-only triangular bursts** on top of `clean`. A burst state machine has `wait` (ticks until the next burst), `n` (burst length in ticks), `amp` and `k` (tick within the burst):

```
draw_next():                                     # noise_rng draws in exactly this order
    wait = round(noise_rng.exponential(burst_gap_s) / DT)
    n    = round(noise_rng.uniform(0.5, 1.5) / DT)
    amp  = burst_peak · noise_rng.uniform(0.5, 1.0) · (+1 if noise_rng.random() < 0.5 else −1)
    k    = 0
at reset (if burst_peak > 0):  draw_next(); wait += round(GRACE_S / DT)      # 30-tick grace
per act:
    if burst_peak == 0 or wait > 0:  (wait −= 1 if wait > 0);  executed = clean;  perturbed = False
    else:
        tri = 1 − |2·(k + 0.5)/n − 1|
        executed = (clamp(clean.steer + amp·tri, −1, 1), clean.throttle);  perturbed = True
        k += 1;  if k ≥ n: draw_next()
```

- **Throttle is never perturbed** (the deadbeat throttle cancels it within one tick).
- At the defaults about 24% of ticks are perturbed, and excursions are about 0.1–0.7 m.

### 5.5 What Step 3 records and trains on

- **Record per tick:** `clean` (label), `executed` (what drove the physics and caused the Frames), `perturbed`. `planned_branch` is optional.
- **Training windows (Step 3 default):** only windows whose 16-tick future has no perturbed ticks. In those windows `executed == clean`, so action targets and future Frames agree exactly, while the observed Frame/State may still be off-line from an earlier burst (recovery data).
- Burst ticks are starting context only, never action targets.

---

## 6. Episode scoring (shared with Step 9)

([Expert API, metrics and CLI](issues/05-expert-api-and-metrics.md))

Policy-agnostic: it uses only the `Info` stream, so it scores the expert and any learned policy identically.

```python
@dataclass(frozen=True)
class EpisodeScore:
    instruction: Instruction
    direction: Direction
    steps: int
    termination: Termination
    crashed: bool
    laps: int
    progress_m: float
    wrong_way_ticks: int
    passages: tuple[Branch, ...]     # every completed Passage, in order
    scored: int
    correct: int
    instruction_followed: bool
    success: bool

def score_episode(track: Track, instruction: Instruction, infos: Sequence[Info]) -> EpisodeScore
def summarize(scores: Sequence[EpisodeScore]) -> dict
```

`infos[0]` is the Info returned by `reset`, and `infos[k]` is the Info after step k.

- `direction = infos[0].direction`; `steps`, `termination`, `laps`, `progress_m` from `infos[-1]`; `crashed = termination == "crash"`.
- `wrong_way_ticks` = number of `infos[1:]` with `wrong_way` True.
- `passages`: for k ≥ 1, whenever `infos[k].passages > infos[k−1].passages`, append `infos[k].last_branch`.
- **Scored passages:** all of `passages`, except the first one when `infos[0].branch is not None` (the episode started inside the fork section, so there was no choice). Partial passages never increment `Info.passages`, so they never appear.
- `correct`: under `outer`/`inner`, scored passages whose branch equals the Instruction; under `any`, `correct = scored`.
- `instruction_followed = (correct == scored) and wrong_way_ticks == 0`.
- `success = not crashed and instruction_followed`.
- `score_episode` raises `ValueError` on an empty `infos`. It needs `track` only for interface stability (future rules may need geometry) and must not depend on it now.
- `summarize` returns a dict with keys `episodes`, `crash_rate`, `laps_mean`, `laps_min`, `instruction_followed_rate`, `success_rate`, `scored_passages`, `wrong_passages`, and `outer_fraction` (over scored passages of `any` episodes, None if there are none).

---

## 7. CLI

```
uv run python -m toycar.expert show [--seed N] [--instruction outer|inner|any] [--direction cw|ccw]
                                    [--clean] [--param KEY=VALUE ...]
uv run python -m toycar.expert eval [--n 1000] [--instruction ...] [--direction ...]
                                    [--clean] [--param KEY=VALUE ...] [--workers K]
```

- `--param` overrides `ExpertParams` fields (float values) and is repeatable. `--clean` sets `burst_peak = 0`. Noise is on by default, because that is what Step 3 records.
- **`show`:**
  - Defaults are seed `int(time.time())` (printed), instruction `any`, direction from the seed.
  - It reuses `toycar.drive`'s window and keys: 64×64 ×8 view, H hi-res, Tab panel, P pause, R next seed, Shift+R same seed, Esc/Q quit. Crash → CRASH banner for 1.5 s → next seed. `max_steps = 10**9`.
  - The panel adds the instruction, planned branch, clean vs executed steer, a `BURST` indicator, `passages`, `last_branch` and `safety_switches`.
- **`eval`:**
  - For each selected (Instruction, Direction), default all 6, it runs seeds `0..n−1` headless with `CarEnv()` (900 ticks) and the §2 recording rule, in parallel (`ProcessPoolExecutor`, `--workers` default = CPU count).
  - It scores each episode with `score_episode`, prints one table row per combination (episodes, crashes, laps/ep, min laps, instruction followed %, scored passages, wrong passages, outer % under `any`, episodes with safety switches) and lists crashing seeds.
  - It prints `PASS` and exits 0 only if there are no crashes and no wrong passages; otherwise it prints `FAIL` and exits 1.
  - Results don't depend on `--workers`.

Expected runtime: about 70 s for `eval --n 1000` (6000 episodes) on an M-series Mac.

---

## 8. Determinism

- Same seed, StartSpec, Instruction, ExpertParams and Track → identical `ExpertAction` sequences and therefore identical episodes, on the same machine with pinned Python/numpy (the Step 1 guarantee).
- Control and rollout: Python `float`/`math` only. Randomness: only the two streams of §5.1, in the fixed draw orders of §5.2 and §5.4.
- The profile cache is keyed by `(track.params, branch, d, a_lat_plan, v_cap)` and is a pure function of the key.

---

## 9. Automated tests (pytest)

| Test | Asserts |
|---|---|
| `test_reliability_clean` | 100 seeds × 6 (Instruction, Direction), `burst_peak = 0`: 0 crashes, all scored passages correct, `instruction_followed` everywhere, laps ≥ 4 per episode, `safety_switches == 0`. |
| `test_reliability_noise` | the same with default params (bursts on). |
| `test_regression_starts` | the seeds that broke earlier rules run without crash, clean and noisy: (838, `any`, cw), (285, `any`, ccw), (3, `outer`, cw/ccw), (26, `inner`, ccw), (81, `any`, cw), (938, `any`, cw). |
| `test_determinism` | two runs of the same (seed, Instruction, params) give identical `ExpertAction` and State sequences; `--clean` vs default noise on `any` give the same `passages` tuple for 20 seeds (separate streams). |
| `test_branch_split` | `any`, 300 seeds × 2 Directions: outer fraction of scored passages in [0.45, 0.55]; with `p_outer = 0.7` in [0.65, 0.75]; `p_outer = 1.0`: every scored passage is outer. |
| `test_noise_shape` | no perturbed tick in the first 30; `executed == clean` whenever not perturbed; throttle never differs; every maximal run of perturbed ticks is at least 8 ticks long (back-to-back bursts may join); `burst_peak = 0` → never perturbed and the noise stream is never drawn. |
| `test_control_pure` | `control` returns identical results on repeated calls and leaves `planned_branch`, `safety_switches` and the RNG states unchanged. |
| `test_profile` | at default params the profile equals `√(5/ρ)` (4.94 m/s) mid-branch-arc and `√(5·7.5)` (6.12 m/s) mid-end-arc to 1e-9, never exceeds `v_cap`, and adjacent samples obey the accel/brake bounds. |
| `test_score_episode` | synthetic Info streams: first passage exempt only when `infos[0].branch` is set; a wrong-branch passage under `outer` gives `instruction_followed = False`; wrong-way ticks counted; `any` counts every passage correct; empty input raises. |
| `test_cli_eval` | `main(["eval", "--n", "3"])` returns 0 and prints `PASS`. |

Full-scale reliability (`eval --n 1000`, clean and default noise) is the manual acceptance in §10, not a pytest.

---

## 10. Manual acceptance (PLAN.md Step 2 check)

1. `uv run python -m toycar.expert eval --n 1000` prints PASS: 0 crashes, 0 wrong passages over 6000 episodes, `any` outer % within 47–53%.
2. `uv run python -m toycar.expert eval --n 1000 --clean` prints PASS.
3. `uv run python -m toycar.expert show --instruction outer --direction cw`, then `inner`/`ccw` and `any`: the car drives sensibly, brakes before curves, takes the instructed branch every time; under `any` it visibly takes both. Bursts show as brief wobbles with `BURST` on the panel, and the car recovers.
4. `show --clean`: the car holds the centerline; panel `passages` increments once per Passage and `last_branch` matches the branch driven.
5. `uv run pytest` passes (Step 1 and Step 2 tests).

---

## 11. Clarifications made while compiling

These fill gaps between ticket decisions. They don't change any decision. If one looks wrong, reopen the relevant ticket.

1. **Order inside `act`:** section-exit redraw (A) → safety rule (B) → clean control (C) → noise (D). Redrawing first means the first tick outside the section already follows the new plan (where both Routes coincide).
2. **Coin draws only under `any`.** Under `outer`/`inner`, `preferred()` returns the Instruction without touching `coin_rng`.
3. **Feasibility rollout** uses the same `control` as driving (pure, no noise) and also checks the start state against `WRONG_BRANCH_M`. If neither branch is feasible, the preferred branch is kept.
4. **Profile grid** has `n = ceil(L/ds)` points, so the last cell is slightly shorter than `ds`; the lookup rounds to the nearest grid index mod n. The profile is built from the expert's Track (not `Track.default()`).
5. **Burst length** `n = round(U[0.5, 1.5] / DT)` gives 8–22 ticks (Python rounds half to even); the triangle uses tick centres (`k + 0.5`), so it never hits exactly 0 or the full peak.
6. **No noise-stream draws when `burst_peak == 0`** (the prototype drew anyway; this spec doesn't, so a clean run has no hidden RNG consumption).
7. **`score_episode` takes `track`** for interface stability only.
8. **`eval` uses the recording rule** (§2) and `CarEnv()` defaults (900 ticks), so its numbers describe the data Step 3 will record.
9. **Exempt is not part of the API.** The prototype's per-passage exempt flag is replaced by the scorer's policy-agnostic rule (§6). It covers the same cases (starts inside the section; partial passages never count).
10. **Prototype sweep numbers** (12,000 episodes each in tickets 03/04, plus 6000-episode edge variants) were produced with the prototype; the implementation must reproduce PASS, not the exact counts.

## 12. Out of scope for Step 2

- Dataset recording format, window cutting, ranges of visual or expert-style randomisation (Step 3).
- Learned models (Steps 4–8) and their closed-loop evaluation (Step 9). Step 9 reuses `score_episode`/`summarize` and may reuse `Expert.control` for relabelling (DAgger).
- Mid-episode Instruction switches (only as a Step 9 evaluation probe).
