# Package layout, API, CLI, outputs and tests

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 03-replanning-scheme.md, 05-metrics-and-gates.md, 06-robustness-report.md
Parent: ../MAP.md

## Question

How is the driver packaged?

- Module(s) (e.g. `wam.drive`), the driver protocol shared by Policy and expert, batched lockstep episodes.
- CLI: `eval <driver>` (verdict, exit code), `show` (pygame window with the model driving, optional imagined future), video/strip export; Run directory `runs/drive/<date>-<name>/` (config, per-episode records, metrics.json, stdout).
- CPU tests with tiny/untrained Policies and the expert (parity with Step 2), runtime < 60 s; manual acceptance steps.

Check: module/API/CLI listing, output files, test list, manual acceptance procedure.

## Comments
- From 03: per-episode-row noise (generator seeded by (episode seed, Tick), sample i offset) bypasses `Policy.act`'s single generator — the driver needs a sampling entry point with per-row initial noise (e.g. `flow.euler` + `flow.guided` with caller-supplied noise, as in `prototypes/replan_variants.py`), or an `act` argument for it. Selection state (previous Chunk) lives in the driver.
- From 04: Evaluation set constants (`dev` `+0…99_999`, `gate` `+100_000 + k` × 120, `robustness` `+200_000 + k`) and the episode list builder (`COMBINATIONS[k % 6]`) live in the driver module; `--episodes N` (multiple of 6) runs a prefix, verdict only at the full gate set (report says "not a gate run" otherwise).
- From 05: CLI surface decided — `eval <driver>` (gate set, verdict only for `wam-v1` at default scheme and full 120; "not gated" / "non-default scheme" / "not a gate run" labels otherwise), `--sweep replan|cfg|samples|steps` on dev seeds `+10_000 + k` × 60 per point, single overrides `--execute-k/--cfg-action/--n-samples/--n-steps`. Drivers needed besides Policies: clean expert and the five Saboteurs (`zero-steer`, `averaged-fork`, `slow` `v_cap` 5, `wrong-branch`, `always-outer`) — tests assert each fails exactly its gate(s) (`prototypes/gate_saboteurs.py`). Report metrics need per-Tick State/action records (distance off centreline, `Expert.control` on visited States, Fork-flip classification of selected Chunks).
- From 06: `eval <driver> --robustness [cond …]` — 9 Stress conditions (`none`, 4 colour Probes, `wide_fork`, `offset_start`, `push`, `push_strong`), 60 robustness seeds each, identical across conditions, own Run directory, never a verdict. Loop needs: per-env Track (wide_fork), RenderParams transform hook (Probes), StartSpec override with redraw (offset_start), and an action-override hook between driver and `env.step` (additive Push, rng `[seed, 4]`; offset start rng `[seed, 5]`). Recovery metrics need Push start/end Ticks in the per-episode record. Definitions and calibration: [Robustness report](06-robustness-report.md).

## Resolution

**Packaging** (grilled with mgrue, 2026-11-05).

**Modules** (`src/wam/`, by concern):

- `wam/drive.py` — Drivers and the closed loop:
  - `Driver` protocol (batched, every Tick): `reset(specs: list[EpisodeSpec], envs: list[CarEnv])`, `act(frames (B,64,64,3) uint8, states: list[State], infos: list[Info]) -> np.ndarray (B, 2)`. Replan timing lives inside the Driver.
  - `Scheme` dataclass: `execute_k=1, n_samples=4, select="coherence", rho=0.9, n_steps=4, cfg_action=1.0, cfg_video=1.0, seed_mode="per_replan"` ([Replanning scheme](03-replanning-scheme.md)).
  - `PolicyDriver(policy, scheme)`: Chunk buffer per row, replans every `execute_k` Ticks, per-row initial noise from a generator seeded by (episode seed, Tick) (sample i offset), coherence selection against the previously selected Chunk, first replan → sample 0; regression objective → `n_samples = 1`.
  - `ExpertDriver(params)` (default for the reference row: `burst_peak = 0`), Saboteur Drivers `zero-steer`, `averaged-fork`, `slow` (`v_cap = 5`), `wrong-branch`, `always-outer` ([Metrics and gates](05-metrics-and-gates.md)).
  - `run_episodes(driver, specs, *, batch=24) -> list[EpisodeRecord]`: lockstep, finished rows drop out, results independent of `batch`. Applies the Stress condition per row: Track, RenderParams transform, StartSpec override with redraw, additive Push between Driver and `env.step` ([Robustness report](06-robustness-report.md)).
  - `EpisodeSpec(seed, instruction, direction, condition)`, `EpisodeRecord(score, fractional_laps, per-Tick arrays, push Ticks, timing)`.
- `wam/closedloop.py` — Evaluation sets (`dev`, `gate`, `robustness`; [Episode protocol](04-episode-protocol.md)), `episode_specs(set, n)`, Stress conditions, metrics, gates, sweeps, report, CLI.
- **API change** in existing code: `flow.sample(…, noise=None)` and `Policy.act(…, noise=None)` accept caller-supplied initial noise `(x_z, x_a)`; default behaviour unchanged.

**CLI** (`python -m wam.closedloop`):

```
eval  <driver> [--name N] [--episodes N] [--execute-k K] [--n-samples N] [--n-steps S] [--cfg-action C] [--cfg-video C]
               [--sweep replan|cfg|samples|steps] [--robustness [COND ...]] [--batch 24] [--device mps|cpu]
show  <driver> [--set gate|dev|robustness] [--k K] [--condition C] [--imagine] [--speed X]
video <run_dir> --episode K [--out DIR] [--hires]
```

- `<driver>`: Checkpoint name or Run directory (→ `PolicyDriver`), `expert`, `saboteur:<name>`.
- `eval` exit codes: 0 PASS or completed without verdict (report states why: "not gated" / "non-default scheme" / "not a gate run"); 1 FAIL (only a `wam-v1` default-scheme full gate run); 2 usage/load error.
- `show`: pygame window (pattern of `toycar.replay.show`): live episode, panel with Tick, Instruction, laps, distance off line, selected Chunk's steer over 16 steps with the other candidates in grey; `--imagine` adds the decoded imagined-future strip.
- `video`: re-renders from stored States (rendering is deterministic, no model) → PNG frames (64 px or 512 px `--hires`) + trajectory PNG over the Track (matplotlib, `viz` group). No mp4 dependency.

**Run directory** `runs/drive/<YYYYMMDD-HHMM>-<name>/`: `config.json` (driver, Checkpoint card sha, Scheme, set, seeds, conditions, git state, device), `episodes.jsonl` (spec, `EpisodeScore`, Fractional laps, Crash Tick/Station, Push Ticks, recovery numbers), `ticks.npz` (per Tick per episode: State x/y/θ/v, executed action, driver action, distance off line, `Expert.control` steer, selected sample index, Fork-flip class), `metrics.json` (summary, gate table + thresholds, CIs, verdict or reason), `report.md` (tables per combination, Δ vs `none` for robustness), `stdout.txt`. Sweeps and robustness: one Run directory, one row group per point/condition.

**CPU tests** (`tests/test_wam_drive.py`, `tests/test_wam_closedloop.py`; tiny Policy + tiny `vae-v1` fixtures; < 60 s total): eval sets (ranges, disjoint, k % 6, balanced prefixes, `--episodes` multiple of 6); expert parity with `toycar.expert.run_episode` (12 seeds, noisy and clean); batch independence (batch 6 vs 1) and determinism; Scheme (replan Ticks 0, k, 2k; coherence picks min-cost on synthetic Chunks; first replan sample 0; regression → 1 sample); noise entry (`act(noise=…)` = `act(seed=…)` for the same noise; `flow.sample` likewise); gates (inclusive thresholds, `None` balance fails, verdict only for full default gate run); Saboteurs on a 24-episode gate prefix fail exactly their gate(s), clean expert passes; Stress conditions (Probe colours, `wide_fork` Track, `offset_start` range/determinism/redraw, Push Ticks, sign stream, driver ≠ executed only during Pushes); metrics (distance off line, Fork-flip class, recovery on synthetic records, Wilson CI values); Run directory + CLI (tiny Policy `--episodes 6` → six files, no absolute paths, exit 0 "not a gate run"; bad args → 2; `video` re-render = frames seen in the run).

**Manual acceptance:**
1. `pytest` green, < 60 s.
2. `eval expert` → all gates PASS (≈ 1 min); `eval saboteur:<each>` → each fails exactly its gate (matches `prototypes/gate_saboteurs.py`).
3. `eval wam-v1` → full 120-episode gate run (≈ 46 min), verdict printed (FAIL is an accepted outcome).
4. `show wam-v1 --imagine` → watched.
5. `eval wam-v1 --sweep replan` (≈ 40 min) and `--robustness` (≈ 3.5 h) finish with reports.
6. Explainer `docs/step9-closed-loop.md` (like `docs/step4-vae.md`, `docs/step5-8-wam.md`): the loop, the Scheme, the gates, `wam-v1`'s results with pictures.
