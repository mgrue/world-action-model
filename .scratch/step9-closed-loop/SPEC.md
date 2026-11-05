# Step 9 spec: closed-loop evaluation — a generic Driver loop, gates for `wam-v1`, sweeps and a robustness report

Buildable spec for PLAN.md Step 9. A **generic closed loop** runs any **Driver** in `toycar`: every `wam.policy` Policy (`wam-v1` and the Step 10 variants), wrapped by the replanning **Scheme**; the clean scripted Expert, as the reference row; and the **Saboteurs** that prove the gates work. It plays fixed **Evaluation sets** of episodes, scores them, writes a report, and gives a **pass/fail verdict for `wam-v1` only**. Step 9 is **evaluation only**: gates are set from what a good driver must achieve, and `wam-v1` failing them is an accepted finding (the cost of imitation, PLAN Step 11).

Sources: every decision here was made in a ticket of `MAP.md`; each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root (Tick, Crash, Lap, Passage, Planned branch, Chunk, Policy, Probe, Evaluation set, Fractional laps, Fork flip, Saboteur, Stress condition, Push, Driver, Scheme, …). Contracts this builds on: `.scratch/step1-simulator/SPEC.md` (`CarEnv`), `.scratch/step2-expert/SPEC.md` (Expert, `score_episode`), `.scratch/step3-dataset/SPEC.md` (seeds, Probes, `sample_render_params`), `.scratch/step5-8-model/SPEC.md` (Policy API §8, Fork references §9.3, Run pattern §10). Prototypes with their outputs in `prototypes/out/` are the reference implementations of the trickier parts: `prototypes/naive_loop.py` (loop, expert parity), `prototypes/replan_variants.py` (Scheme, noise, selection, Fork flips), `prototypes/gate_saboteurs.py` (gates), `prototypes/stress_calibrate.py` (Stress conditions). Section 16 lists clarifications made while compiling.

PLAN.md check: "Each tick: feed current frame, state and instruction. Execute the first few predicted actions, then replan. Metrics: laps completed, crashes, instruction followed." Measured by the four gates of §6 on the gate set (§3), with the loop of §4.

---

## 1. Scope and layout

([ticket 07](issues/07-api-cli-tests.md))

```
src/wam/drive.py        Driver protocol, Scheme, PolicyDriver, ExpertDriver, Saboteur Drivers,
                        EpisodeSpec / EpisodeRecord, run_episodes (lockstep, batched), Stress-condition application
src/wam/closedloop.py   Evaluation sets, episode_specs, Stress conditions table, metrics, gates, verdict, sweeps,
                        report, Run directory, the CLI (python -m wam.closedloop)
src/wam/flow.py         CHANGED: sample(..., noise=None)                (§11)
src/wam/policy.py       CHANGED: Policy.act(..., noise=None)            (§11)
tests/test_wam_drive.py, tests/test_wam_closedloop.py
docs/step9-closed-loop.md   explainer (§14)
```

- No changes to `toycar` (simulator, Expert, dataset), to Checkpoints or to training ([MAP](MAP.md) Out of scope).
- Reuse: `wam/runs.py` (Run directories, git state, hashes, stdout tee), `wam.evaluate.fork_references` / `classify` / `MAX_ENTRY` / `COMMIT_R`, `toycar.expert.score_episode` / `summarize` / `start_spec`, `toycar.dataset.CLOSED_LOOP_SEED_BLOCK` / `COMBINATIONS` / `PROBES` / `sample_render_params`.

## 2. Environment and budget

([tickets 01](issues/01-does-it-drive.md), [03](issues/03-replanning-scheme.md), [04](issues/04-episode-protocol.md))

- Apple M1 Pro, 16 GB, MPS; Python 3.14, torch 2.15. Default device MPS, else CPU. Tests CPU only.
- The simulator waits for the model: no real-time constraint.
- Measured costs (MPS): `act` 64 ms at B = 1; batched 19–25 ms per Chunk at B = 8–60. Default Scheme ≈ 23 s per episode lockstep-batched (24 episodes ≈ 9.3 min) → **≈ 46 min per 120-episode gate run**. Expert / Saboteur episodes: ≈ 0.5 s each on CPU (120 in ≈ 70 s, no rendering needed for scoring).

---

## 3. Episodes (Evaluation sets)

([ticket 04](issues/04-episode-protocol.md))

### 3.1 Evaluation sets

Fixed seed ranges inside `CLOSED_LOOP_SEED_BLOCK = 9_000_000`, defined once as constants in `wam/closedloop.py`:

| Set | Seeds | Count | Use |
|---|---|---|---|
| `dev` | `+0 … 99_999` | free | tuning, prototypes; sweeps use `+10_000 + k`, 60 per point (§7) |
| `gate` | `+100_000 + k` | k = 0…119 | the only seeds a verdict is computed on; never tuned on |
| `robustness` | `+200_000 + k` | k = 0…59 | every Stress condition (§8) |

- Episode k of a set plays `COMBINATIONS[k % 6]` (Instruction × Direction, `toycar.dataset` order: outer-cw, outer-ccw, inner-cw, inner-ccw, any-cw, any-ccw). Every prefix whose length is a multiple of 6 is balanced.
- `episode_specs(set, n, condition="none") -> list[EpisodeSpec]`. `n` must be a multiple of 6 (else `ValueError`) and ≤ the set's count (dev: ≤ 90_000 from `+10_000`).
- **Every Driver runs the identical list** → paired rows.
- Dev seeds already used in prototypes: `+0…59`, `+1000…1023`, `+10_000…` are free.

### 3.2 One episode

- `env = CarEnv(track=…)` (default Track unless the Stress condition changes it), `max_steps = 900` (60 s at 15 Hz, ≈ 4 expert Laps).
- `env.reset(seed, start_spec(instruction, direction), render_params=sample_render_params(seed))` — the Step 3 start rule (outer/inner start on their own Route, `any` random) and colour jitter; the start pose's lateral offset ±1 m, heading ±10°, speed 0–6 m/s are drawn by the env from `default_rng(seed)`.
- A Crash ends the episode. First replan at Tick 0, no warm-up.
- Scoring: `score_episode(track, instruction, infos)` over the Info stream (first Passage exempt when starting inside the fork section), plus Fractional laps = `infos[-1].progress_m / track.length_m`.

---

## 4. Drivers and the loop

([tickets 02](issues/02-chunk-execution-practice.md), [03](issues/03-replanning-scheme.md), [05](issues/05-metrics-and-gates.md), [07](issues/07-api-cli-tests.md))

### 4.1 Driver protocol

```python
class Driver(Protocol):
    name: str
    def reset(self, specs: list[EpisodeSpec], envs: list[CarEnv], states: list[State], infos: list[Info]) -> None
    def act(self, rows: list[int], frames: np.ndarray, states: list[State], infos: list[Info]) -> np.ndarray
        # rows: indices into the reset() lists of the episodes still running, in increasing order;
        # frames uint8 (len(rows), 64, 64, 3) (None for Drivers with needs_frames = False);
        # returns float (len(rows), 2) actions, one per running episode, every Tick
    needs_frames: bool
```

- `act` is called once per Tick for all running episodes; the Driver decides internally when to replan. The loop treats Expert and Policies identically.
- Optional per-Tick diagnostics a Driver may expose after `act` (`last_diag: dict[str, np.ndarray]` keyed per row): `PolicyDriver` exposes `replanned` (bool), `selected` (sample index), `chunk_steer` (H,) of the selected Chunk when it replanned. Used for Fork flips (§5) and `show`.

### 4.2 Scheme and `PolicyDriver`

```python
@dataclass(frozen=True)
class Scheme:
    execute_k: int = 1          # actions executed per replan, 1 ≤ execute_k ≤ H
    n_samples: int = 4
    select: str = "coherence"   # "coherence" | "first"
    rho: float = 0.9
    n_steps: int = 4
    cfg_action: float = 1.0
    cfg_video: float = 1.0
    seed_mode: str = "per_replan"  # "per_replan" | "per_episode"
DEFAULT_SCHEME = Scheme()
```

`PolicyDriver(policy, scheme)`, per running row j with episode seed `s_j`:

1. **Replan** when its action queue is empty (at Tick 0, execute_k, 2·execute_k, …): all rows needing a replan this Tick go into one batched sampling call (in sub-batches of `batch` rows, §4.4).
2. **Initial noise, per row and sample** (independent of batching): sample i of row j uses a CPU `torch.Generator().manual_seed(g)` with `g = (s_j · 64 + i) · 100_003 + tick` for `per_replan`, `g = s_j · 64 + i` for `per_episode`; `x_z, x_a = flow.initial_noise(cfg, 1, generator)`; concatenated in row-major (row, sample) order and passed as `noise=` to `Policy.act` (§11). (Exactly the prototype's rule, `replan_variants.py`.)
3. **Sampling:** `policy.act(frames, features, instructions, n_samples=N, n_steps, cfg_action, cfg_video, noise=…)` → clipped actions `(rows, N, H, 2)`. Regression-objective Policies force `N = 1` (and `cfg = 1`).
4. **Selection:** the first replan of an episode, or `select == "first"`, takes sample 0. Otherwise `coherence`: with the previously *selected* Chunk `prev` and overlap `ov = H − execute_k`, keep the candidate minimising `Σ_{τ<ov} ρ^τ · ‖a[τ] − prev[execute_k + τ]‖₂` (2-D action norm); ties → lowest index. If `ov = 0`, take sample 0.
5. **Execute** the first `execute_k` actions of the selected Chunk, one per Tick.
6. **Never** average Chunks or samples (no ensembling offered).

### 4.3 `ExpertDriver` and Saboteurs

- `ExpertDriver(params=ExpertParams(burst_peak=0.0))`: one `Expert(env.track, instruction, params)` per row, `reset(seed, state, info)`, `act(state, info).executed` each Tick. The **clean** expert is the reference row (the upper bound); `ExpertDriver(ExpertParams())` (noisy, as recorded) is used only by the parity test.
- Saboteurs, `saboteur:<name>` — the clean expert with one defect ([ticket 05](issues/05-metrics-and-gates.md)):

| Name | Construction | Must fail |
|---|---|---|
| `zero-steer` | expert throttle, steer 0 | Crashes, Laps (and balance: no scored `any` Passage) |
| `averaged-fork` | in `any` episodes, action = mean of `control(state, "outer", d)` and `control(state, "inner", d)` every Tick (outside the fork section the two coincide) | Crashes (Island) |
| `slow` | `ExpertParams(burst_peak=0, v_cap=5.0)` | Laps only |
| `wrong-branch` | expert told the opposite Branch under `outer`/`inner` (`any` unchanged); scored under the true Instruction | Instruction followed only |
| `always-outer` | `ExpertParams(burst_peak=0, p_outer=1.0)` | `any` balance only |

The expert Driver's `act` must call `Expert.act` every Tick (it advances the coin/plan state) even when the action is overridden.

### 4.4 `run_episodes`

```python
@dataclass(frozen=True)
class EpisodeSpec:
    seed: int; instruction: Instruction; direction: Direction; condition: str = "none"

def run_episodes(driver: Driver, specs: list[EpisodeSpec], *, batch: int = 24, shadow_expert: bool = True,
                 progress: Callable | None = None) -> list[EpisodeRecord]
```

- Creates one env per spec, applies the Stress condition (§8.1: Track, RenderParams transform, StartSpec override, Push), resets, `driver.reset`, then steps all running episodes in **lockstep**: each Tick render Frames (if `needs_frames`), `driver.act`, apply Push (driver action → executed action), `env.step`. Finished episodes drop out.
- `batch` bounds the rows per sampling call (24 rows × `n_samples`); results must not depend on `batch` (per-row noise, §4.2).
- `shadow_expert`: a clean `Expert` per row, fed the same States/Infos via `act` every Tick (to advance its plan), gives `expert_steer = control(state, planned_branch, direction)[0]` for the off-distribution action error (§5).
- `EpisodeRecord`: `spec`, `score: EpisodeScore`, `fractional_laps`, `crash_tick` / `crash_station_m` / `crash_in_section` (None if no Crash), `push_ticks: list[int]` (Push start Ticks), per-Tick arrays over `steps` Ticks — `x, y, theta, v` (State after the step, plus the start State at index 0: length steps + 1), `action_driver (steps,2)`, `action_exec (steps,2)`, `dist_m (steps,)` (`track.distance_to_centerline` after the step), `branch (steps+1,)` (−1 / 0 outer / 1 inner from `info.branch`), `station_m (steps+1,)`, `expert_steer (steps,)`, `selected (steps,)` (−1 when no replan), plus the replans list `(tick, chunk_steer)` for Policies; `wall_s`, `ms_per_replan`.

---

## 5. Metrics

([ticket 05](issues/05-metrics-and-gates.md); report-only unless listed in §6)

Computed per combination (6 rows) and in total, for every Driver, by `metrics(records) -> dict`:

| Group | Metric |
|---|---|
| Gate inputs | `crash_rate`, mean Fractional laps, `instruction_followed_rate`, `outer_fraction` (from `summarize` + Fractional laps), each rate with its 95 % Wilson interval |
| Outcome | `success_rate`, integer `laps_mean` / `laps_min` (PLAN wording), Fractional laps of non-crashed episodes, mean lap time (s) = 900 / 15 / Fractional laps over non-crashed episodes |
| Crashes | per episode Crash Tick and Station, in fork section or not; Station histogram (2 m bins) |
| Line holding | `dist_m` over all Ticks: mean, p95, max |
| Off-distribution action error | RMSE of `action_exec[:,0]` (before Push: `action_driver[:,0]`) vs `expert_steer`, binned by `dist_m` < 0.3 / 0.3–0.6 / 0.6–1.0 / > 1.0 m; in `any` episodes Ticks inside the fork section or within `MAX_ENTRY` Ticks before an entry are excluded (the shadow expert's coin may pick the other Branch) |
| Fork | `wrong_passages`, Passages per Branch per combination, Island Crashes (Crash with `crash_in_section`), **Fork flips** (Policies only) |
| Smoothness | mean \|Δ `action_driver` steer\| per Tick, total wrong-way Ticks |
| Cost | ms per replan, wall time |

**Fork flips** ([ticket 03](issues/03-replanning-scheme.md); `replan_variants.fork_flips`): for each entry Tick `te` into the fork section (`branch[t] ≠ −1`, `branch[t−1] = −1`), take every replan at Tick `t` with `1 ≤ e = te − t ≤ min(MAX_ENTRY, H)`; classify its selected Chunk's steer, multiplied by +1 (cw) / −1 (ccw), with `evaluate.classify(steer, ref[e,0], ref[e,1])` against `ref = fork_references(open_dataset("data/step3-v1", check=False), horizon=H)`. An approach with ≥ 2 classified replans flips if the classes disagree. Reported: approaches, flips, `any` approaches, `any` flips, fraction uncommitted (`r ≥ COMMIT_R`). n/a for the expert, Saboteurs and the `wide_fork` Track.

Wilson interval (z = 1.96): `c = (p + z²/2n)/(1 + z²/n)`, `h = z·sqrt(p(1−p)/n + z²/4n²)/(1 + z²/n)`, clipped to [0, 1].

## 6. Gates and verdict

([ticket 05](issues/05-metrics-and-gates.md))

| Gate | Definition | Threshold | Clean expert (gate set) |
|---|---|---|---|
| Crashes | `crash_rate` | **≤ 0.05** | 0 % |
| Laps | mean Fractional laps | **≥ 3.5** | 4.12 |
| Instruction followed | `instruction_followed_rate` | **≥ 0.95** | 100 % |
| `any` balance | `outer_fraction` | **0.30 ≤ x ≤ 0.70**; `None` → FAIL | 0.43 |

- Thresholds inclusive. Absolute, from what a good driver does; not from `wam-v1`'s numbers.
- **Verdict** (`PASS` iff all four pass) only when **all** hold: driver is the Policy `wam-v1` (Checkpoint name), Scheme = `DEFAULT_SCHEME`, set = `gate`, all 120 episodes, no Stress condition. Otherwise the gate table is printed with a reason instead of a verdict: `not gated` (any other Driver), `non-default scheme`, `not a gate run` (`--episodes` < 120).
- Precision at 120 episodes (95 % Wilson): observed 0 % → 0–3.1 %; 5 % → 2.3–10.5 %; 25 % → 18.1–33.4 %.
- Integer laps is not the gate: `floor(progress / 88.06 m)` scores a crash-free driver at 97 % of expert pace as 3 Laps everywhere.
- **Validation** (`prototypes/gate_saboteurs.py`, 120 gate seeds): clean expert PASS; `zero-steer` 100 % crash, 0.05 Fractional laps; `averaged-fork` 33 % crash (all 40 `any`), 2.89; `slow` (`v_cap` 5) 3.39 → laps only; `slow` `v_cap` 6 / 7 → 3.85 / 3.99 PASS (85–97 % pace is fine); `wrong-branch` 33 % followed → followed only; `always-outer` outer 1.00 → balance only.

## 7. Sweeps

([ticket 05](issues/05-metrics-and-gates.md))

- `eval <driver> --sweep <name>`: dev seeds `+10_000 + k`, 60 episodes per grid point (10 per combination), one report row group per point, never a verdict. The clean expert reference row runs once on the same 60 seeds.
- Grids (other Scheme fields at default):

| Name | Grid | ≈ cost (MPS) |
|---|---|---|
| `replan` | `execute_k` ∈ {1, 2, 4, 8, 16} (values > H skipped) | 40 min |
| `cfg` | `cfg_action` ∈ {1, 1.5, 2, 3} | 1.5–2 h |
| `samples` | `n_samples` ∈ {1, 4, 8} | 1 h |
| `steps` | `n_steps` ∈ {2, 4, 10} | 1 h |

- Single overrides on any run: `--execute-k --n-samples --n-steps --cfg-action --cfg-video --select --rho --seed-mode`.

## 8. Robustness report

([ticket 06](issues/06-robustness-report.md)) — report-only, never a gate or exit code, for any Driver.

### 8.1 Stress conditions

Applied by `run_episodes` per spec; everything else as §3.2.

| Condition | Produced by |
|---|---|
| `none` | unstressed, the paired baseline |
| `brown_grass`, `light_road`, `blue_car`, `no_kerb` | `render_params = PROBES[name].apply(sample_render_params(seed))` |
| `wide_fork` | `CarEnv(track=Track(PROBES["wide_fork"].track_params))` (`bulge_m = 7`); Fork flips n/a |
| `offset_start` | `StartSpec(direction=d, route=start_spec(...).route or random, station_m=U(0, L), lateral_m=±U(1.0, 1.5), heading_offset=±U(10°, 20°) (rad), speed=U(0, 6))`, all draws from `default_rng([seed, 5])` in this order: route (`any` only, `integers(2)`), station, lateral sign (`choice([−1, 1])`), lateral magnitude, heading sign, heading magnitude, speed; redraw (≤ 50 tries) while `env.reset` raises `ValueError` |
| `push` | from Tick 60, every 60 Ticks (60, 120, …): for 8 Ticks, `exec_steer = clip(driver_steer + sign · 0.4, −1, 1)`, throttle unchanged; one sign per Push from `default_rng([seed, 4]).choice([−1, 1])`; the Driver is not told |
| `push_strong` | the same with 0.5 |

Pushes are additive by design: an absolute steer override (±0.8 for 6 Ticks) crashed even the clean expert within 4 Ticks in arcs (corners sit 1.52 m from the centreline at zero offset; the limit is 2.8 m). Calibration (`prototypes/stress_calibrate.py`, clean expert, 60 robustness seeds): `none`, `wide_fork`, additive 0.3–0.5 × 8 → 0 % crashes; 0.6 × 10 → 100 % (the cliff). `offset_start` 1.0–1.5 m / 10–20° → ≈ 12 % expert crashes (physically lost starts).

### 8.2 Runs and metrics

- `eval <driver> --robustness [COND …]` (all 9 by default): robustness seeds k = 0…59, **identical across conditions** → paired; clean expert reference rows for every condition (≈ 1 min total). ≈ 23 min per condition → ≈ 3.5 h for one Policy.
- The full §5 table per condition (gates shown "report-only") and a "Δ vs `none`" column, plus:
  - `push*`: crash within 45 Ticks (3 s) of a Push start; peak `dist_m` in those 45 Ticks; Ticks from Push end until `dist_m < 0.3` (censored at the next Push or episode end; report median and fraction recovered).
  - `offset_start`: crash rate on the episodes the clean expert survives (next to the plain rate); Ticks until first `dist_m < 0.3`.
- 95 % CI at 60 episodes ≈ ±11 pp at 25 %.

---

## 9. Run directory `runs/drive/<YYYYMMDD-HHMM>-<name>/`

([ticket 07](issues/07-api-cli-tests.md))

| File | Contents |
|---|---|
| `config.json` | driver name/kind, Checkpoint name + card sha256 (Policies), Scheme, mode (`gate` / `sweep:<name>` / `robustness`), set, seeds, conditions, `batch`, device, git commit + dirty flag, start time, argv |
| `episodes.jsonl` | one line per episode (and per sweep point / condition, which it carries): spec, Driver row (`driver`, `point`), `EpisodeScore` fields, Fractional laps, Crash Tick/Station/in-section, Push Ticks, recovery numbers |
| `ticks.npz` | the per-Tick arrays of §4.4, concatenated over episodes, with an `offsets` array (start index per episode, Tick axis) and `offsets_state` for the steps + 1 arrays |
| `metrics.json` | per Driver row: §5 metrics, gate table with thresholds and CIs, `verdict` (`PASS` / `FAIL`) or `no_verdict_reason` |
| `report.md` | human-readable tables: gate table, per-combination table, reference expert row and paired Δ, sweep / robustness row groups |
| `stdout.txt` | tee of the console (`runs.py`) |

Name defaults to the driver name (plus `-sweep-<name>` / `-robustness`); clashes get a numeric suffix. Repo-relative paths only.

## 10. CLI (`python -m wam.closedloop`)

([ticket 07](issues/07-api-cli-tests.md))

```
eval  <driver> [--name N] [--episodes N] [--execute-k K] [--n-samples N] [--n-steps S] [--cfg-action C] [--cfg-video C]
               [--select first|coherence] [--rho R] [--seed-mode per_replan|per_episode]
               [--sweep replan|cfg|samples|steps] [--robustness [COND ...]] [--batch 24] [--device mps|cpu] [--json PATH]
show  <driver> [--set gate|dev|robustness] [--k K] [--condition C] [--imagine] [--speed X] [scheme overrides]
video <run_dir> --episode K [--point P] [--out DIR] [--hires]
```

- `<driver>`: a Checkpoint name or Run directory → `PolicyDriver(wam.policy.load(...), scheme)`; `expert` → clean `ExpertDriver`; `saboteur:<name>`.
- `eval` runs the selected Driver **and** the clean expert reference on the same specs; prints the gate table, the per-combination table and the verdict or reason. `--sweep` and `--robustness` are mutually exclusive.
- **Exit codes:** 0 = PASS, or completed without a verdict; 1 = FAIL (only a `wam-v1` default-Scheme full gate run); 2 = usage or load error (argparse, unknown driver/condition, `--episodes` not a multiple of 6).
- `show`: pygame window (pattern of `toycar.replay.show`, 512 × 512 view + side panel): one live episode (`--set`, `--k` pick the spec; default gate k = 0), panel with Tick, Instruction, Direction, laps, distance off line, the selected Chunk's steer over H steps with the other candidates in grey; `--imagine` adds a strip of the selected candidate's decoded imagined Frames (`return_latents=True`, `policy.compressor.decode`). Keys: Space pause, N next episode, Esc/Q quit. Not tested automatically.
- `video`: re-renders episode K of a Run from `ticks.npz` States with `toycar.render.render(state, track, params, resolution)` (deterministic; no model) → `frame_00000.png …` (64 px, or 512 px with `--hires`) and `trajectory.png` (path over the Track, Crash marked; matplotlib from the `viz` group). Default `--out` = `<run_dir>/video/<K>/`.

## 11. Changes to `wam.flow` / `wam.policy`

([ticket 03](issues/03-replanning-scheme.md), [07](issues/07-api-cli-tests.md))

```python
def sample(model, cfg, obs, state, instr, *, n_steps=10, cfg_action=1.0, cfg_video=1.0, generator=None,
           noise: tuple[Tensor | None, Tensor] | None = None)   # flow.sample
def act(self, frame, state, instruction, *, n_samples=1, n_steps=10, cfg_action=1.0, cfg_video=1.0, seed=None,
        return_latents=False, noise: tuple[Tensor | None, Tensor] | None = None) -> ActResult   # Policy.act
```

- `noise = (x_z, x_a)` with shapes `(B·n_samples, K, 16, 8, 8)` (None without video) and `(B·n_samples, H, 2)` in row-major (row, sample) order, as `flow.initial_noise` returns; when given, it replaces the draw from `generator` / `seed` (passing both raises `ValueError`); shape mismatch raises `ValueError`. Ignored by the regression objective. Default (`None`) behaviour unchanged; Steps 5–8 tests stay green.

---

## 12. Automated tests (pytest, CPU only, < 60 s total)

`tests/test_wam_drive.py`, `tests/test_wam_closedloop.py`, using the tiny Policy and tiny `vae-v1` fixtures from `tests/conftest.py`:

| Test | Asserts |
|---|---|
| `test_eval_sets` | seed ranges and disjointness; combination `k % 6`; prefixes of multiples of 6 balanced; `n` not a multiple of 6 → `ValueError` |
| `test_expert_parity` | `ExpertDriver` through `run_episodes` gives `EpisodeScore`s identical to `toycar.expert.run_episode` on 12 seeds, noisy (`ExpertParams()`) and clean |
| `test_batch_independence` | tiny Policy, 6 episodes (max_steps shortened via the env), `batch=6` vs `batch=1` → identical records; two runs → identical |
| `test_scheme_replan_ticks` | a stub Policy counting calls: replans exactly at Ticks 0, k, 2k, … for k ∈ {1, 4} |
| `test_coherence_select` | synthetic candidates: picks the min-cost one (ρ weighting, overlap H − k), ties → lowest index, first replan → sample 0, `select="first"` → 0 |
| `test_regression_single_sample` | regression-objective tiny Policy → one sample per replan |
| `test_noise_entry` | `Policy.act(noise=…)` equals `act(seed=g)` when the noise is `initial_noise` from the same generator; `flow.sample(noise=…)` likewise; both given → `ValueError`; wrong shape → `ValueError` |
| `test_gates` | inclusive thresholds at the boundaries; `outer_fraction = None` fails balance; verdict only for (wam-v1, default Scheme, gate, 120, `none`), else the right reason |
| `test_saboteurs` | on a 24-episode gate-set prefix: each Saboteur fails exactly its gate(s) of §4.3, the clean expert passes all four |
| `test_conditions` | Probe colours applied to the jittered params; `wide_fork` Track in use; `offset_start` within ranges, deterministic, redrawn when invalid; Push active exactly on Ticks [60 + 60m, 68 + 60m) with signs from `[seed, 4]`; `action_driver ≠ action_exec` only on Push Ticks |
| `test_metrics` | `dist_m` stats, Fork-flip classification and recovery metrics on synthetic records; Wilson intervals vs known values (0/120 → 0–3.1 %, 30/120 → 18.1–33.4 %) |
| `test_cli_run_dir` | `main(["eval", <tiny run dir>, "--episodes", "6", "--device", "cpu"])` → exit 0, report says "not a gate run"/"not gated", all six files exist, no absolute paths; bad args → 2 |
| `test_video_rerender` | `video` re-render of a recorded episode equals the Frames the Driver saw during the run |

## 13. Performance targets

- Default Scheme on MPS: ≤ 30 s per episode lockstep-batched (measured ≈ 23 s); full gate run ≤ 60 min.
- Expert / Saboteur gate run (120 episodes): ≤ 2 min CPU.
- Memory: batch 24 × 4 samples within the Steps 5–8 inference footprint (no new target).

## 14. Manual acceptance (PLAN Step 9)

1. `uv run pytest` green, new tests < 60 s.
2. `python -m wam.closedloop eval expert` → all four gates pass (≈ 1 min); `eval saboteur:<each>` → each fails exactly its gate(s), matching §6 Validation.
3. `python -m wam.closedloop eval wam-v1` → full 120-episode gate run (≈ 46 min), gate table and verdict printed, exit code matches. FAIL is an accepted outcome (dev estimate ≈ 25 % crashes, Fractional laps ≈ 3.1).
4. `python -m wam.closedloop show wam-v1 --imagine` → watched: the car drives, the panel's Chunk and candidates update every Tick.
5. `eval wam-v1 --sweep replan` (≈ 40 min) and `eval wam-v1 --robustness` (≈ 3.5 h) finish with reports.
6. Explainer `docs/step9-closed-loop.md` (style of `docs/step4-vae.md`, `docs/step5-8-wam.md`): the loop and why replanning is the model's only feedback; the Scheme and why Chunks are never averaged; the gates and the Saboteurs; `wam-v1`'s gate, sweep and robustness results with pictures (`video` trajectories, Crash Station histogram, action error by distance off line).

## 15. Hand-offs to later steps

- **Step 10 experiments** are one command each: Policy variants via `eval <variant-run-or-checkpoint>` (same gate set, "not gated", expert reference row); replanning rate via `--sweep replan`; CFG via `--sweep cfg`; small world changes via `--robustness`. Variants with H < 16: `execute_k ≤ H`, coherence overlap `H − execute_k`, Fork references truncated to H.
- **Step 11:** the clean expert row and the paired Δ in every report are the comparison with classical control.
- **Follow-up map** (out of this spec): teaching the model to recover (DART-style demonstration noise, DAgger, fork oversampling) → `wam-v2`, evaluated with this loop unchanged.

---

## 16. Clarifications made while compiling

These fill gaps between ticket decisions without changing any decision. If one looks wrong, reopen the relevant ticket.

1. **The expert reference row runs inside every `eval`** on the same specs (≈ 1 min), so every report carries its paired Δ; ticket 05 asked for "the expert row as reference" in every table.
2. **Noise seed formula** `(seed · 64 + i) · 100_003 + tick` is the one used by every dev measurement in ticket 03 (`replan_variants.py`), kept so results are reproducible against the prototype.
3. **Shadow expert for the action error** (as in `prototypes/diag_crash.py`): its `any` coin can disagree with the Branch the Driver takes, so `any` Ticks near and in the fork section are excluded from that metric.
4. **Fork references** are built once per `eval` from `data/step3-v1` (the `wam-v1` training data) with `horizon = H` of the Policy. A hidden `--data PATH` option (Steps 5–8 hidden-option pattern) points it at the tiny dataset in tests.
5. **Coherence with `ov = 0`** (execute_k = H) degenerates to sample 0; the `replan` sweep at k = 16 therefore equals `select="first"`.
6. **`--episodes`** applies to `eval` gate runs only; sweeps and robustness have fixed counts (60).
7. **Batch size 24** is the measured lockstep default; `--batch` changes speed only.
8. **Push timing** starts at Tick 60 and repeats every 60 Ticks (4 s) to episode end; recovery is censored at the next Push.
9. **`offset_start` draw order** (§8.1) is fixed here so seeds are reproducible; it is the order of `prototypes/stress_calibrate.py`.
10. **Lap time** is derived from Fractional laps over non-crashed episodes (60 s / Fractional laps), not from per-Lap Tick stamps.

## 17. Out of scope

- Retraining or new data (DAgger, DART, fork oversampling) → separate follow-up map (`wam-v2`). Step 9 only evaluates.
- The Step 10 experiment analyses and the Step 11 write-up themselves (only made one command each).
- Real-time / wall-clock constraints.
- Changing the simulator, Expert, dataset or Checkpoints.
- RTC-style inpainting, PAINT noise inversion, best of 8 as default (ticket 03 / 05 decisions), mp4 export.
