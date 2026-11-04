# Package layout, API, CLI, tests and acceptance

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 03-transformer-blocks.md, 05-training-recipe.md, 06-sampler.md, 07-evaluation-and-checks.md
Parent: ../MAP.md

## Question

What modules, API and commands deliver the model, and what proves them correct?

- Modules in `src/wam/` by concern (`model.py`, `train.py`, `sample.py`, `evaluate.py`?), reuse of Step 4 helpers (Runs, promote/load, provenance).
- Policy API for Step 9: `load_policy(name)` → `act(frame, state, instruction, *, seed) -> actions (16, 2)` (+ optional imagined Frames); batched variant for Step 10.
- Checkpoint `checkpoints/wam-v1/` with card (config, compressor name `vae-v1`, dataset manifest hash, metrics).
- CLI: `train`, `eval`, `sample` / `show`, `promote` — mirroring `wam.compressor`.
- Automated CPU tests (tiny model, tiny dataset): token packing shapes and positions, noising formula, loss masks per Step 10 switch, determinism, resume, sampler shapes, API round trip.
- Manual acceptance: the PLAN checks for Steps 7 and 8.

Check: API and CLI sketch with signatures, the list of automated tests, and the manual acceptance steps.

## Comments
- From 04: the Policy API takes and returns raw actions in [−1, 1] (de-normalise + clip inside); training switches to expose: `objective`, `video_stream`, `video_loss_weight`, `action_loss_weight`, `t_shift`, `horizon` (+ 02's `future_stride`, `p_uncond`), with the cross-checks (regression ⇒ no video; w_v = 0 ⇒ no video). Replanning rate is an inference argument, not a training switch.
- From 03: the prototype `prototypes/model_proto.py` uses `einops` (not a project dependency; run with `uv run --with einops`). Implementation: add `einops` as a dependency or use plain reshapes — decide here. CPU tiny arch (d64×2) runs forward/backward of all switch combinations in seconds; zero-init heads give exactly 0 at init (testable).
- From 05: CLI mirrors Step 4 — `train` (config, `--steps`, `--resume`, `--name`), `eval`, `show`, `promote`; Runs under `runs/wam/`; Checkpoints `checkpoints/wam-v1[-variant]/` with `card.json`; `checkpoint.pt` = EMA weights + arch + switches + action normalisation. Tests: resume bit-identity (20 == 10 + 10, CPU tiny), deterministic val.
- From 06: `wam.policy.load` / `Policy.act` / `Policy.sample` signatures fixed (see 06 Resolution). Tests: same seed → identical Chunk; CFG rejected for regression; output shape (B, n_samples, H, 2), values in [−1, 1]; zero-init model → actions = de-normalised Euler of pure noise path (velocity 0 ⇒ actions = clipped de-normalised initial noise).
- From 07: `eval` CLI (gate table, exit 0 / 1, `--json`); frozen eval data: Fork reference Chunks (16 entry offsets × 2 Branches × 16) and the fixed subsets (970 general, 512 Fork end-of-Run, 192 + 64 periodic) — where they live (module constant vs file) is decided here. Tests: the metric functions on fakes (labels pass, averaged references fail commit, shuffled fail following), on a tiny val set.
- Grilled 2026-11-04 (Q1–Q4); template `.scratch/step4-vae/SPEC.md` §1, §8–10. The Step 3 tiny fixture (`tests/conftest.py`: 12/6/6 episodes × 60 Ticks) has no reliable Fork Windows, so Fork-metric tests use synthetic arrays plus one real-data test.
- Q1 layout: `src/wam/runs.py` (new; `_git_info`, `_sha256`, `_rel`, `_now`, resume-file handling extracted from `compressor.py`, which imports them — behaviour unchanged, guarded by Step 4 tests); `model.py` (`ModelConfig` = arch + every switch + action-normalisation constants, validated; position ids; patchify; RoPE; Block; LastLayer; WAM); `flow.py` (action (de)normalisation, t sampling + shift, noising, losses, Euler + CFG; pure maths, no I/O); `policy.py` (`Policy`, `load`, `act`, `sample`, `promote`; CLI `python -m wam.policy`); `train.py` (sampler, EMA, val, logging, resume); `evaluate.py` (subsets, Fork references, metrics, gates, contact sheet). Add `einops` to the `model` dependency group.
- Q2 CLI: `train [--name] [--steps 32000 | --quick] [--seed 0] [--data data/step3-v1] [--set key=value …] [--cpu]`, `train --resume runs/wam/<run>`, `eval <run | checkpoint> [--json OUT] [--sweeps]` (exit 0 PASS / 1 FAIL), `show <run | checkpoint> [--out samples.png]` (needs viz group, else exit 2), `promote runs/wam/<run> --name wam-v1[-variant] [--force]`. `--quick` = 3,200 steps; switches only via `--set`, validated by `ModelConfig` (invalid → exit 2); `train` ends with `eval` + `show`, exit 0 iff Step 7 check and 07 gates pass; `--sweeps` adds CFG and `n_steps` sweeps (≈ 25 min, not run by `train`). Promote copies the EMA `checkpoint.pt` → `checkpoints/<name>/model.pt` + `card.json` (config incl. switches and normalisation constants, compressor name + card hash, dataset manifest hash, git sha + dirty, hash of `src/wam/*.py`, metrics + thresholds used, resumes, created); refuses an existing name, an unfinished Run, a failing Run without `--force`; no absolute paths. Frozen eval data computed, not stored: `evaluate.fork_references(ds)`, `evaluate.subsets(ds)` deterministic from the dataset and fixed seeds; their hash goes into `metrics.json` and the card.
- Q3 automated tests (CPU, < 60 s added, tiny arch d64×2, tiny fixture), `tests/test_wam_{model,flow,policy,train,evaluate}.py`:
  1. position ids = 02's table (162); counts 290 / 90 / 54 / 34 (stride 1, H 8, H 4, no video)
  2. patchify ↔ unpatchify exact; token i = cell (i // 4, i % 4), channels `(c ph pw)`
  3. default arch exactly 8,229,888 params
  4. zero-init heads output exactly 0
  5. `ModelConfig` rejects regression + video, w_v = 0 + video, H ∉ {4, 8, 16}, stride ∉ {1, 2}, d % 64 ≠ 0
  6. RoPE relative: a common t shift leaves outputs unchanged
  7. token permutation → permuted outputs
  8. noising x_t = t·ε + (1−t)·x, target ε − x; t ∈ (0, 1), median = shift(0.5) = 0.6226
  9. action normalisation round trip, constants
  10. loss = weighted per-Stream means; no-video has no video term; regression target a_n; `p_uncond` ≈ 0.1 on 10k
  11. Euler with constant-velocity fake = noise − v exactly; forward count = `n_steps`; schedule 1 → 0 shifted
  12. CFG: w = 1 one pass; w ≠ 1 formula and 2B batch; regression + CFG raises
  13. `act`: (B, n_samples, H, 2) float32 in [−1, 1]; B = 1 kept; Instruction name / int; same seed identical, different differs; `return_latents` shape; zero-init model → clip(denorm(initial noise))
  14. promote / load round trip; refusals (existing name, unfinished, failing without `--force`); code-hash mismatch warns; no absolute paths in the card
  15. same seed → equal weights, different seed → different
  16. 20 steps == 10 + resume + 10 bit for bit (incl. EMA and optimiser)
  17. EMA update and warm-up formula; weight decay on Linear weights only
  18. deterministic val (two calls equal)
  19. metrics on synthetic Chunks: labels pass, averaged references fail commit, shuffled fail following, one-Branch collapse fails two-sided; general RMSE of labels = 0
  20. subsets and their hash deterministic
  21. CLI smoke: tiny `train --steps 5` → `eval` exit 1 → `promote --force` → `show` (skip without matplotlib); `--set objective=regression --set video_stream=false` trains 2 steps; invalid `--set` exits 2
  22. real data (skip if `data/step3-v1` absent): Fork references and fakes reproduce 07 (labels commit ≥ 0.999, average 0), arrays only
  23. real Checkpoint (skip if `checkpoints/wam-v1` absent): `load` + `act` on 8 val Windows, shapes and range
  The existing `toycar` torch-free test stays.
- Q4 manual acceptance: (1) `uv sync && uv run pytest` green incl. real-data tests; (2) `caffeinate -i uv run python -m wam.policy train --quick --name quick` (≈ 20 min): Step 7 line PASS; any one-time threshold tightening happens now, in a commit editing SPEC.md citing the quick Run's numbers; (3) `caffeinate -i uv run python -m wam.policy train --name v1` (≈ 3.4 h), exit 0; (4) `samples.png`: samples lie on labels, the `any` Fork panel shows two bundles (one per reference), imagined Frames show the car moving along the Route; (5) `promote runs/wam/<run> --name wam-v1`, commit `checkpoints/wam-v1/` (≈ 33 MB fp32); (6) `eval wam-v1` same verdict, exit 0. PLAN Step 7 = separate `loss_video` / `loss_action` (+ val) columns and eval's Step 7 PASS; PLAN Step 8 = eval gate table PASS. Step 10 variants: promote with `--force`, commit only when a Step 10 result cites them.

**Resolution (2026-11-04):** modules `wam.{runs, model, flow, policy, train, evaluate}` (+ `einops`), CLI `python -m wam.policy {train, eval, show, promote}` with `--set` switches, `checkpoints/wam-v1/` = EMA `model.pt` + `card.json`, eval data computed with recorded hash, 23 automated tests, 6 manual acceptance steps — all as above.

