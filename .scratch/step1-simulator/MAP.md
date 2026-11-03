# Step 1 simulator spec

Labels: wayfinder:map
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step1-simulator/SPEC.md`: a buildable spec for PLAN.md Step 1 (deterministic 2D overhead car simulator, 64×64 RGB renderer, keyboard driving mode) that an agent can implement without follow-up questions — **including the interface contract** Steps 2–4 rely on (state/action definitions, seed & reset semantics, headless rendering, track centerline/lanes/direction, progress & lap info).

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `CONVERSATION_SUMMARY.md` at repo root). Runs on a 16 GB Apple Silicon Mac.
- Skills: `/grilling` + `/domain-modeling` (glossary in repo-root `CONTEXT.md`, created lazily), `/prototype` for the look/feel ticket, `/research` subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research findings, under `research/`) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket). Progress/laps are part of the Env API ticket.
- Planning only: no simulator code beyond throwaway prototypes.

## Decisions so far

<!-- one line per closed ticket -->

## Not yet specified

- **Visual randomisation knobs** (Step 3 "minor visuals"): which parameters the renderer exposes (track/grass colours, car colour, noise?) and their ranges — depends on the look settled in the geometry prototype and how much Step 4's VAE should be stressed.
- **Determinism scope**: bit-exact across machines / numpy versions, or only same-machine? May need float64-only rules or a frame-hash golden file; sharpens once stack and physics are fixed.
- **Speed observability**: a single frame can't show speed (summary flags this); whether the sim/renderer should help (e.g., motion trail, history) or leave it to the model's state token — revisit after physics + render are fixed.

## Out of scope

- Step 2 scripted expert, Step 3 dataset recording, Step 4 VAE — only their *needs* from the sim are in scope, via the interface contract.
- Implementing the simulator itself (that's what the spec hands off).
