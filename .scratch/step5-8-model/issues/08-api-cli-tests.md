# Package layout, API, CLI, tests and acceptance

Labels: wayfinder:grilling
Status: open
Assignee:
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
