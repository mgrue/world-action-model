# Expert variation knobs and demonstration noise

Labels: wayfinder:grilling
Status: open
Assignee: mgrue
Blocked by: 01-controller-research.md, 02-instruction-set.md, 03-controller-prototype.md
Parent: ../MAP.md

## Question

A single deterministic expert produces a narrow data distribution; the learned policy will drift off it in closed loop (Step 9). Which variation does the expert itself expose, and how is it seeded?

- Style knobs per episode: target-speed scale, lateral offset preference, lookahead/gain jitter? Which are allowed without breaking the reliability check?
- Demonstration noise: inject noise into the *executed* action while recording the clean expert action (DART), or record the noisy action? Noise model (Gaussian per tick, coloured/held, occasional larger perturbations) — knobs only; ranges belong to Step 3.
- Determinism: the expert's own RNG (seeded how, relative to the `reset` seed), so an episode is reproducible from (seed, Instruction, knobs).
- Which actions does Step 3 store: executed, expert-clean, or both?

Check: every knob is listed with its meaning, default, safe range from the prototype, and the RNG it draws from; the reliability check still passes at the edges of the safe ranges.

## Comments
