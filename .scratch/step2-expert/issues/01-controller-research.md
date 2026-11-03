# Path-tracking controllers and demonstration noise for imitation learning

Labels: wayfinder:research
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

For a kinematic-bicycle car following a known centerline of lines and arcs (tightest radius 4.875 m, 15 Hz, instant steering, lateral-acceleration clamp, no reverse), which path-tracking law is the best fit: pure pursuit (fixed vs speed-scaled lookahead), Stanley, or curvature feed-forward + feedback? What are the standard ways to plan a speed profile from curvature with a braking horizon?

Second half: how do imitation-learning setups make expert data robust to compounding error — DART-style noise injection (perturb the *executed* action, record the clean expert action), DAgger, randomised starts — and what noise scales are typical? Which of these fit a fully scripted expert with a deterministic simulator?

Check: a short findings file under `research/` with a recommended controller family, a speed-profile method, and a recommendation (with rationale) on demonstration noise, each with sources.

## Comments
**Resolution (2026-11-03):** Findings in [research/01-controller-and-demo-noise.md](../research/01-controller-and-demo-noise.md), from the literature plus about 100–200-episode experiments in the real simulator.
- **Controller:** curvature feed-forward plus lateral and heading feedback (the Samson / Paden 2016 rear-wheel law), **not pure pursuit** (PLAN.md's "pure pursuit" is treated as a placeholder). The sim's reference point turns with exactly the commanded curvature and steering is instant, so feed-forward is exact: about 1 mm RMS error against 0.13–0.55 m for pure pursuit. Feed forward the *mean* curvature over the next tick's arc (handles the S-bend sign flips). Gains ke = 0.25 m⁻², kθ = 1.0 m⁻¹ (critically damped, about 4 m to settle, speed-independent). Clamp the curvature using the speed after this tick's throttle.
- **Route to follow = the Planned branch, never the nearest projection:** near the Fork and Merge, nearest-projection caused the only clean-run crashes.
- **Speed profile:** v_lim = √(a_lat_plan/|κ|), then forward (4 m/s²) and backward (6 m/s²) passes on a 0.05 m station grid, twice around the loop, per Route × Direction; deadbeat throttle to the profile 0.5 m ahead. **a_lat_plan = 5 m/s²** (≈ 4.9 / 6.1 m/s on branch / end arcs, ≈ 4.1 laps per episode). The margin matters more than the gains, because the lateral clamp also limits how much feedback can correct.
- **Demo noise (input to ticket 04):** noise is needed, since the clean expert sits on the centerline after about 4 m. Execute noisy, record the clean label, and record the executed action too (DART). Use sparse triangular steer bursts (about 1 s, peak ≤ 0.4, about 25% of the time): 0 crashes, 0.3–0.7 m excursions. Continuous correlated (OU) noise crashed more often at similar RMS. Emit a per-tick `perturbed` flag so Step 3 can pick windows with clean futures. The crash budget is about 1 m of lateral error once heading error is included. DAgger isn't needed now, but keep the expert a pure function of (State, Instruction, branch memory) so it can label any state later.
