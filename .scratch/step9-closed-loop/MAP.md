# Step 9 closed-loop evaluation spec

Labels: wayfinder:map
Status: open
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: ../MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step9-closed-loop/SPEC.md`: a buildable spec for PLAN.md Step 9 — a **generic closed-loop driver** that runs any driver in `toycar` (every `wam.policy` Policy: `wam-v1` and the Step 10 variants; and the scripted expert through the identical loop, as the reference row), with episode protocol, metrics, report and a **pass/fail verdict for `wam-v1` only** — implementable by an agent without follow-up questions.

PLAN.md check to satisfy, turned into measurable gates: "Each tick: feed current frame, state and instruction. Execute the first few predicted actions, then replan. Metrics: laps completed, crashes, instruction followed."

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `README.md`, explainers `docs/step4-vae.md`, `docs/step5-8-wam.md`). Builds on implemented Steps 1–8 (`.scratch/step{1..4}-*/SPEC.md`, `.scratch/step5-8-model/SPEC.md`).
- Vocabulary follows repo-root `CONTEXT.md` (Tick, Crash, Lap, Passage, Planned branch, Chunk, Policy, Probe, …); new terms are added there as tickets resolve.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` where a question needs measuring, research subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research and throwaway prototypes) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no driver code in `src/` beyond throwaway prototypes in `prototypes/`.
- Facts that constrain the driver:
  - `wam-v1` is promoted (`checkpoints/wam-v1/`): open-loop gates all PASS (rmse steer 0.064 / throttle 0.019, commit 0.99, follow 1.0, `any` balance 0.47, two-sided 0.996; Run `runs/wam/20261104-1928-v1`). `wam.policy.load("wam-v1")` → `Policy.act(frame, state.features(), instruction, *, n_samples, n_steps=10, cfg_action, cfg_video, seed, return_latents)` → clipped actions (B, n_samples, 16, 2). 66 ms per Chunk at B = 1 on MPS (≈ 70 ms per replan incl. encode), 21.5 ms per Chunk batched. Replan rate was explicitly left to Step 9 (Steps 5–8 SPEC §15).
  - Simulator: `toycar.env.CarEnv` (15 Hz Ticks, `max_steps` 900 = 60 s, Crash ends the episode), `env.render()` → 64 × 64 Frame; `reset(seed, start_spec, render_params)`. Expert ≈ 4.1 laps per 900-Tick episode.
  - Scoring exists: `toycar.expert.score_episode(track, instruction, infos)` / `summarize` (crash, laps, progress, wrong-way Ticks, Passages, Instruction followed, success, `any` outer fraction); first Passage exempt when starting inside the fork section.
  - Seeds: `toycar.dataset.CLOSED_LOOP_SEED_BLOCK = 9_000_000` is reserved for Step 9 and never recorded. Episode streams: env `default_rng(seed)`, expert coin `[seed, 1]`, expert noise `[seed, 2]`, colours `sample_render_params(seed)` = `[seed, 3]`.
  - Probes (Step 3 §3.2): colour Probes `brown_grass`, `light_road`, `blue_car`, `no_kerb` (already degrade `vae-v1`), and `wide_fork` (Track change, closed-loop only).
  - Known weak spots (Steps 5–8 SPEC §15): off-Route Frames (thin recovery data), the Fork commit; mild imagined-future drift with k.
  - Hardware: Apple M1 Pro, 16 GB, MPS; Python 3.14, torch 2.15.
  - Conventions to reuse: `src/wam/` by concern, Runs under `runs/<kind>/<date>-<name>/` with config/metrics.json/stdout, CPU tests with tiny models < 60 s, exit 0 = PASS.

## Decisions so far

<!-- one line per closed ticket -->
- [How chunked policies are executed in practice](issues/02-chunk-execution-practice.md) — execute 25–50 % of a Chunk then replan; never average Chunks (it picks the Island at the Fork); keep branches across replans training-free via shared noise or best-of-N backward-coherence selection (BID), with RTC-style inpainting as a fallback

## Not yet specified

- **If `wam-v1` fails the closed-loop gates:** which remedies stay inside this spec (loop/sampler-side: replan rate, sample selection, CFG, ensembling) and at what point the answer becomes "retrain" (out of scope). Sharpens once the prototype and the replanning ticket have numbers.

## Out of scope

- Retraining or new data for the model (DAgger, noise-injected recordings, fork oversampling) — a new map if Step 9 shows it is needed.
- The Step 10 experiment analyses and the Step 11 comparison write-up themselves — this spec only makes them one command each.
- Real-time / wall-clock constraints: the simulator waits for the model.
- Changing the simulator, expert, dataset or Checkpoints.
