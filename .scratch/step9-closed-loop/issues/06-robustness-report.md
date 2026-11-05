# Robustness report

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 05-metrics-and-gates.md
Parent: ../MAP.md

## Question

Which report-only stress tests run, and how?

- Colour Probes (`brown_grass`, `light_road`, `blue_car`, `no_kerb`) and the `wide_fork` Track.
- Starts away from the expert's distribution (lateral offset, heading error, speed), and recovery after a push (state perturbation mid-episode).
- Same metrics as 05; episode counts; whether any of it is ever a gate (default: no).

Check: the list of stress conditions, how each is produced through `CarEnv`/`RenderParams`, counts and runtime.

## Comments
- From 04: robustness episodes use seeds `CLOSED_LOOP_SEED_BLOCK + 200_000 + k` (combination `k % 6`), same episode rule as the gate set (900 Ticks, Step 3 start rule) except the stressed factor. Budget reference: ≈ 23 s per episode for the default scheme → ≈ 46 min per 120 episodes; 60 per condition gives ±≈ 11 pp at 25 %.
- From 05: report the same metric table as the gate report (gates shown as "report-only", never a verdict), incl. Fractional laps, Crash Stations, distance off the centreline and action error vs `Expert.control` binned by distance — the last two are the natural readouts for off-distribution starts and pushes. Run the clean expert under every stress condition as the reference row (it must stay crash-free under colour Probes trivially; under pushes/offset starts it shows what is recoverable at all).

## Resolution

**Robustness report** (grilled with mgrue, 2026-11-05). Report-only, never a gate or exit code, for any driver.

**Stress conditions** (9; everything else as the gate episode: Step 3 start rule, `sample_render_params(seed)` jitter, 900 Ticks, default scheme):

| Condition | Produced by |
|---|---|
| `none` | unstressed — the paired baseline |
| `brown_grass`, `light_road`, `blue_car`, `no_kerb` | `PROBES[name].apply(sample_render_params(seed))` |
| `wide_fork` | `CarEnv(track=Track(PROBES["wide_fork"].track_params))` (`bulge_m = 7`); Fork flips n/a (references are for the default Track) |
| `offset_start` | `StartSpec(direction, route per start_spec (random for any), station U(0, L), lateral ±U(1.0, 1.5) m, heading ±U(10°, 20°), speed U(0, 6))` drawn from rng `[seed, 5]`; redraw (≤ 50) while `reset` rejects the pose |
| `push` | from Tick 60, every 60 Ticks: add ±0.4 to the executed steer for 8 Ticks (clip to [−1, 1]); sign from rng `[seed, 4]`; the driver is not told and replans from the State it finds |
| `push_strong` | the same with ±0.5 for 8 Ticks |

**Calibration** (`prototypes/stress_calibrate.py`, clean expert, 60 robustness seeds, 35 s CPU): `none` / `wide_fork` / `push` ±0.3–0.5 × 8 → 0 % crashes, 4.06–4.12 Fractional laps; ±0.6 × 10 and ±0.8 × 10 → 100 % (cliff: the push alone puts a corner off the road — corners sit 1.52 m from the centreline at zero offset, limit 2.8 m); an *absolute* steer override (±0.8 × 6) crashed the expert within 4 Ticks in arcs, hence additive Pushes. Offset starts just outside the gate's start range (±1 m, ±10°) are partly unrecoverable even for the expert: 1.0–1.5 m / 10–20° → 11.7 %, 1.0–1.25 m / 10–15° → 10 %.

**Metrics:** the full gate-report table (gates shown "report-only"), a "Δ vs `none`" column (paired seeds), plus:
- `push*`: crash within 45 Ticks (3 s) of a Push start; peak distance off the centreline in those 45 Ticks; Ticks from Push end until back within 0.3 m (censored at the next Push).
- `offset_start`: crash rate **on episodes the clean expert survives** (recoverable starts only), next to the plain rate; Ticks until first within 0.3 m.
- Expert reference rows for every condition (≈ 1 min total).

**Counts and runtime:** 60 episodes per condition, robustness seeds `CLOSED_LOOP_SEED_BLOCK + 200_000 + k`, k = 0…59 (combination k % 6), **identical seeds for every condition** → paired. 95 % CI ≈ ±11 pp at 25 %. ≈ 23 min per condition → ≈ 3.5 h for all 9 for one Policy (overnight).

**Command:** `eval <driver> --robustness [cond …]` (all 9 by default), its own Run directory and report.

Assets: [`prototypes/stress_calibrate.py`](../prototypes/stress_calibrate.py); outputs [`prototypes/out/stress_calibrate_stdout.txt`](../prototypes/out/stress_calibrate_stdout.txt), `stress_calibrate.json`.
