# Step 2 scripted expert spec

Labels: wayfinder:map
Tracker: local markdown — tickets are files in `issues/`, each with `Parent: MAP.md`.
Ticket header fields: `Labels:` (wayfinder type), `Status:` (open/closed), `Assignee:` (empty = unclaimed — the claim), `Blocked by:` (ticket files; unblocked when all are closed).
Frontier = open, unblocked, unassigned tickets in `issues/`.

## Destination

`.scratch/step2-expert/SPEC.md`: a buildable spec for PLAN.md Step 2 (scripted expert driving the `toycar` simulator under a small set of Instructions) that an agent can implement without follow-up questions — **including the interface contract** Steps 3 and 9 rely on (Instruction set and labels, expert API, branch-choice semantics that create genuinely multimodal situations, variation knobs the dataset can sample, and the instruction-following / lap metrics reused by closed-loop evaluation).

PLAN.md check to satisfy: "the expert completes laps reliably for every behavior."

## Notes

- Domain: learning project — toy world action model mirroring FLUX 3 Action (see `PLAN.md`, `CONVERSATION_SUMMARY.md` at repo root). Builds on the accepted Step 1 simulator: `.scratch/step1-simulator/SPEC.md`, code in `src/toycar/`.
- Vocabulary follows repo-root `CONTEXT.md`; new terms (Instruction, Expert, …) are added there as tickets resolve.
- Skills: `/grilling` + `/domain-modeling` for decision tickets, `/prototype` for the controller ticket, research subagents for research tickets.
- Git: single local repo, **no branches** — commit everything (incl. research findings under `research/`, throwaway prototypes under `prototypes/`) to `master`.
- Each ticket carries its own acceptance check (no separate testing ticket).
- Planning only: no expert code in `src/` beyond throwaway prototypes.
- Facts from Step 1 that constrain the expert: 15 Hz; instant steer; lateral-accel clamp (curve speed limits ≈ 6.24 m/s on branch arcs ρ = 4.875 m, ≈ 7.75 m/s on end arcs R = 7.5 m); brake 8 m/s², accel 4 m/s², V_MAX 10; no reverse, so the expert can never recover by backing up; starts are U[±1 m] lateral, U[±10°] heading, U[0, 6] m/s, anywhere on either Route; Crash = any corner > 2.8 m from both centerlines; episodes are 900 ticks by default.

## Decisions so far

<!-- one line per closed ticket -->

- [Instruction set and branch choice](issues/02-instruction-set.md) — three branch-only Instructions `outer`/`inner`/`any` (Direction stays an episode property); fair coin per Fork passage under `any`, drawn at reset and on each exit from the fork section; fixed per episode; starts match the Instruction, wrong-branch passages exempt; `INSTRUCTIONS` tuple order frozen; CFG null is model-side only

## Not yet specified

_(empty — mid-episode changes and the CFG null label were resolved in [Instruction set and branch choice](issues/02-instruction-set.md))_

## Out of scope

- Step 3 dataset recording format, window cutting and the *ranges* of visual or expert-style randomisation — only the knobs the expert exposes are in scope.
- Learned models (Steps 4–8) and closed-loop evaluation of them (Step 9) — only the metric definitions Step 9 reuses are in scope.
- Changes to the Step 1 simulator, unless a ticket finds a blocking gap (then reopen the relevant Step 1 ticket).
- Implementing the expert itself (that's what the spec hands off).
