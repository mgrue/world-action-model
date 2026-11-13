# Step 10a JEPA encoder spec

Labels: wayfinder:map
Status: open
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: ../MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step10a-jepa/SPEC.md`: a buildable spec for the PLAN.md §10a encoder **`jepa-v1`** (I-JEPA style: context encoder, EMA target encoder, predictor discarded after training, car-biased masking, no pixel reconstruction), trained on recorded Frames and promoted as a frozen Checkpoint. An agent must be able to implement it without follow-up questions. It **includes the interface contract** the later WAM switch and comparison rely on: feature-grid shape and per-channel normalisation, an `encode`/`load` API aligned with `wam.compressor` (no `decode`), Checkpoint card, and the linear probe as a reusable component that scores both `vae-v1` Latents and `jepa-v1` features.

PLAN.md check to satisfy, made measurable: "a linear probe on frozen features reads car x, y and heading" at the bar `vae-v1` set, plus "no collapse".

## Notes

- Domain: learning project, a toy world action model mirroring FLUX 3 Action (see `PLAN.md` §10a, `README.md`, explainers `docs/step4-vae.md`, `docs/step5-8-wam.md`, `docs/step9-closed-loop.md`). Builds on implemented Steps 1–9. The closest model is the Step 4 compressor: `.scratch/step4-vae/SPEC.md` and `MAP.md`, `src/wam/compressor.py`.
- Vocabulary follows repo-root `CONTEXT.md` (Frame, State, Latent, Checkpoint, Probe, Split, Run, …). New terms (e.g. JEPA feature grid, linear probe) are added there as tickets resolve. "Probe" already means a colour/Track world change; the linear probe needs a name that does not clash.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` where a question needs measuring, research subagents for research tickets.
- Git: single local repo, **no branches**. Commit everything (incl. research under `research/`, throwaway prototypes under `prototypes/`) to `master`, with `Claim: <ticket> (Step 10a)` / `Resolve: <ticket> (Step 10a)` commits.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no encoder code in `src/` beyond throwaway prototypes in `prototypes/`.
- Scope decisions from charting: **I-JEPA** (each Frame alone, no temporal prediction); **no speed** in the probe (a single Frame cannot show it).
- Facts that constrain the encoder:
  - Frames: 64 × 64 × 3 uint8, re-rendered from stored States (`frame_dataset`, `FrameSampler` in `wam.compressor`); `data/step3-v1`, 6000 episodes × 301 Frames, train 4800 episodes. Fixed Track with per-episode colour jitter; the only moving thing is the car, ≈ 7 × 4 px (0.7 % of the image), red body + yellow nose. One 8 × 8-pixel patch = 5 × 5 m. Exact State of every Frame known (privileged labels for the probe and for car-biased masking).
  - `vae-v1` (`checkpoints/vae-v1/`): 8 × 8 × 16 Latent, per-channel normalised, 3.17 M params, ≈ 1 h 25 min training. It passed the Step 4 check through a **pose estimator on decoded pictures**: position p95 0.11 m / max 0.30 m, heading p95 ≈ 8° (bar: pos p95 ≤ 0.3, max ≤ 0.6 m; head p95 ≤ 10°, max ≤ 30°). No linear probe on its Latents exists yet.
  - WAM token layout (`src/wam/model.py` `patchify`): a Frame's grid is grouped 2 × 2 into 16 tokens; `vae-v1` gives 64 values per token, a 32-channel JEPA grid would give 128. Only the picture input/output layers widen.
  - Colour Probes `brown_grass`, `light_road`, `blue_car`, `no_kerb` already degrade `vae-v1` heavily; report-only here too.
  - Hardware: Apple M1 Pro, 16 GB, MPS; Python 3.14, torch 2.15, float32 eager, `num_workers=0`; seeded MPS runs bit-identical.
  - Conventions to reuse: `src/wam/` by concern, Runs under `runs/<kind>/<date>-<name>/`, Checkpoints under `checkpoints/<name>/` with `card.json` (code hash, provenance), never overwritten; CPU tests with tiny models < 60 s; CLI exit 0 = PASS.

## Decisions so far

<!-- one line per closed ticket -->

## Not yet specified

- **Training data beyond on-Route Frames.** Whether JEPA (no pixel loss) needs more variety than `frame_dataset` gives (off-Route Frames, extra colour augmentation), e.g. for the colour Probes. Depends on what the architecture prototypes and probe show.
- **Reference numbers for the later comparison map.** Which `jepa-v1` diagnostics the comparison will want recorded on the card (e.g. probe on Probes, per-situation slices), beyond the gate itself.
- **Explainer.** A `docs/step10a-jepa.md` like `docs/step4-vae.md`; shape depends on what the spec ends up containing.

## Out of scope

- The WAM switch `future_target=vae|jepa`, training the JEPA-target world models, and the comparison runs (action gates, closed loop, imagined-future probes). A follow-up map once `jepa-v1` exists and passes its gate.
- V-JEPA / temporal prediction across Frames. Decided at charting: I-JEPA only.
- Probing speed: invisible in a single Frame.
- The pure JEPA world model with planning by search (PLAN §10a "Later, separate idea").
- Changing the simulator, dataset, `vae-v1` or `wam-v1`.
