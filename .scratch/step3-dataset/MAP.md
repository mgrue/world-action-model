# Step 3 dataset spec

Labels: wayfinder:map
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step3-dataset/SPEC.md`: a buildable spec for PLAN.md Step 3 (record a few thousand expert episodes and cut them into training windows) that an agent can implement without follow-up questions — **including the interface contract** Steps 4–10 rely on: the on-disk episode format, the window definition and its time alignment, the train/val/test split, the randomisation ranges (start, speed, minor visuals) and what is held out for the Step 10 generalisation experiments, and a loader that yields windows without depending on the model code.

PLAN.md check to satisfy: "plotting a window shows frames and actions that line up in time."

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `CONVERSATION_SUMMARY.md` at repo root). Builds on the accepted Step 1 simulator (`.scratch/step1-simulator/SPEC.md`) and the implemented Step 2 expert (`.scratch/step2-expert/SPEC.md`, `src/toycar/expert.py`).
- Vocabulary follows repo-root `CONTEXT.md`; new terms (Window, Split, …) are added there as tickets resolve.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` where a question needs measuring, research subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research findings under `research/`, throwaway prototypes under `prototypes/`) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no dataset code in `src/` beyond throwaway prototypes.
- Facts from Steps 1–2 that constrain the dataset:
  - 15 Hz, 900-tick episodes by default (60 s, about 4 laps, about 3.7 Passages); a window spans 17 Frames ≈ 1.13 s ≈ 6–11 m of travel.
  - The expert yields per tick `clean` (label), `executed` (drove the physics) and `perturbed`; Step 2 decided the default training windows are those whose 16-tick future has no perturbed tick (about 24% of ticks are perturbed at default noise).
  - Recording rule: `StartSpec(route=instruction)` for `outer`/`inner`; `any` keeps a random start route. Episodes are reproducible from (seed, StartSpec, Instruction, ExpertParams, RenderParams, TrackParams).
  - `reset` already randomises start pose: any Station on either Route, lateral U[±1 m], heading U[±10°], speed U[0, 6] m/s.
  - `RenderParams` exposes grass/road/kerb/body/nose colours and supersampling; `CarEnv.reset(..., render_params=...)` sets them per episode. Track shape is `TrackParams`.
  - `a_lat_plan`/`v_cap` are hidden per-episode style: Step 2 says keep them fixed unless varying them is the experiment.
  - Measured (M-series Mac): expert + physics ≈ 0.08 ms/tick, render ≈ 0.07 ms/Frame. One 900-tick episode is 11 MB of raw uint8 Frames, ≈ 0.16 MB with `np.savez_compressed`. 3000 episodes ≈ 33 GB raw, ≈ 0.5 GB compressed; re-rendering Frames from stored States is also cheap.
  - Hardware: 16 GB Apple Silicon Mac; training (Steps 4–8) uses PyTorch on MPS.

## Decisions so far

<!-- one line per closed ticket -->

- [Storage format and loading throughput](issues/01-storage-and-loading.md) — store States/actions/flags/metadata only and re-render Frames in the loader (0.11 GB at 3000 episodes, bit-identical, 3.5k windows/s vs about 13/s consumed by training); per-episode Frame hash + render-code hash in the manifest, checked on open; HDF5+lzf as fallback; raw Frames (33 GB) don't fit
- [Episode mix, dataset size and splits](issues/02-episode-mix-and-splits.md) — 6000 episodes × 300 ticks, exactly 1000 per Instruction × Direction (assigned `k mod 6`); Step 1 start ranges and Step 2 expert defaults unchanged; by-episode split 4800/600/600 on seed blocks 1M/2M/3M (Step 9 eval 9M, Step 2 eval 0–999); any crash aborts recording; ≈ 0.98M usable windows
- [Window definition and time alignment](issues/04-window-definition.md) — window at t = fₜ + sₜ, `clean` aₜ…aₜ₊₁₅, future fₜ₊₁…fₜ₊₁₆ (action k ↔ future Frame k), Instruction id; clean-future filter by default, stride 1, explicit index, fixed 2048-window val/test subsets; `horizon` and `future_frames` are loader settings (Step 6 token budget); per-tick station/branch/planned_branch stored; `approaches_section` tag; uint8 HWC Frames, no torch in core
- [Minor-visual randomisation and held-out world changes](issues/03-visual-randomisation.md) — per-episode jitter of all five colours (×U[0.85, 1.15] brightness, ±10 per channel, stream `[seed, 3]`), same for all splits; no pixel augmentation; track shape fixed; five reserved probes (`brown_grass`, `light_road`, `blue_car`, `no_kerb`, `wide_fork`); Step 1 renderer made recolourable (bit-identical, 0.1 ms per new colour set)

## Not yet specified

_(empty — Frame history deferred to Step 6 as a loader option and the Step 6 token budget handled by the `horizon`/`future_frames` loader settings, both in [Window definition](issues/04-window-definition.md); latent caching moved to Out of scope)_

## Out of scope

- Caching VAE latents next to the data (Step 4/6 storage decision; 8×8×16 float16 latents are 2 KB per Frame, about 3.7 GB for the 6000 × 301 Frames decided in ticket 02; see [ticket 01](research/01-storage-and-loading.md)).
- The models (Steps 4–8): architecture, normalisation inside the model, batching on MPS beyond what the loader contract needs.
- Closed-loop evaluation (Step 9) — only the seed ranges it must not overlap are in scope.
- Changes to the Step 1 simulator or Step 2 expert, unless a ticket finds a blocking gap (then reopen the relevant ticket).
- Implementing the recorder and loader (that's what the spec hands off).
