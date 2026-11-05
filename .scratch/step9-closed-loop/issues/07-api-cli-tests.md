# Package layout, API, CLI, outputs and tests

Labels: wayfinder:grilling
Status: open
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
