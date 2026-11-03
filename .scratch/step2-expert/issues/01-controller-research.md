# Path-tracking controllers and demonstration noise for imitation learning

Labels: wayfinder:research
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

For a kinematic-bicycle car following a known centerline of lines and arcs (tightest radius 4.875 m, 15 Hz, instant steering, lateral-acceleration clamp, no reverse), which path-tracking law is the best fit: pure pursuit (fixed vs speed-scaled lookahead), Stanley, or curvature feed-forward + feedback? What are the standard ways to plan a speed profile from curvature with a braking horizon?

Second half: how do imitation-learning setups make expert data robust to compounding error — DART-style noise injection (perturb the *executed* action, record the clean expert action), DAgger, randomised starts — and what noise scales are typical? Which of these fit a fully scripted expert with a deterministic simulator?

Check: a short findings file under `research/` with a recommended controller family, a speed-profile method, and a recommendation (with rationale) on demonstration noise, each with sources.

## Comments
