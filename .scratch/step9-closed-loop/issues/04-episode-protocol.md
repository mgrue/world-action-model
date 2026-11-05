# Episode protocol

Labels: wayfinder:grilling
Status: closed
Assignee: mgrue
Blocked by: 01-does-it-drive.md
Parent: ../MAP.md

## Question

Which episodes is a driver evaluated on?

- Seeds from `CLOSED_LOOP_SEED_BLOCK` (dev subset for tuning vs held-out gate set), episodes per Instruction × Direction, episode length (900 Ticks?), start rule (Step 2/3 `start_spec`), colour jitter (`sample_render_params`).
- How many episodes give stable estimates (variance from ticket 01), and the time budget per full evaluation.
- Same seeds for every driver (expert and Policies) so rows are paired.

Check: the seed list rule, counts, and expected runtime; confidence intervals for the headline rates at that count.

## Comments
- From 01: per-episode outcomes are very variable at the current skill level (crash ticks 19–822); 60 episodes at execute 4 took 121 s wall, 24 at execute 1 took 420 s (lockstep, MPS). Expert parity holds with the Step 3 start rule + `sample_render_params`.
- From 03: default scheme costs ≈ 9.3 min per 24 episodes (lockstep batched, MPS; ≈ 23 s per episode); plain execute-1 first-sample ≈ 6.4 min. Dev seeds `CLOSED_LOOP_SEED_BLOCK + 1000…1023` were used for tuning (and 0…59 in 01): the gate set must be disjoint. Crash counts at 24 episodes vary 3–8 between near-identical schemes — size the gate set for that.

## Resolution

**Episode protocol** (grilled with mgrue, 2026-11-05):

- **Evaluation sets** (new `CONTEXT.md` term), fixed seed ranges inside `CLOSED_LOOP_SEED_BLOCK = 9_000_000`, defined once as constants in the driver:

  | Set | Seeds | Use |
  |---|---|---|
  | `dev` | `+0 … 99_999` | tuning, sweeps, prototypes (already holds 0…59 and 1000…1023) |
  | `gate` | `+100_000 + k`, k = 0…119 | the only seeds a verdict is computed on; never tuned on |
  | `robustness` | `+200_000 + k` | Probes, off-distribution starts, pushes (counts → [Robustness report](06-robustness-report.md)) |

- **Combination:** episode k of a set plays `COMBINATIONS[k % 6]` (Instruction × Direction, interleaved), so every prefix of a multiple of 6 is balanced. Every driver (expert, `wam-v1`, Step 10 variants) runs the identical list → paired rows. Policy noise stays seeded by (episode seed, Tick) ([Replanning scheme](03-replanning-scheme.md)), so results do not depend on batching.
- **Episode:** `env.reset(seed, start_spec(instruction, direction), render_params=sample_render_params(seed))` — the Step 3 start rule and colour jitter (expert parity checked in [Does wam-v1 drive at all?](01-does-it-drive.md)); `max_steps = 900` Ticks (60 s, ≈ 4 expert laps); a Crash ends the episode; first replan at Tick 0, no warm-up.
- **Count:** 120 gate episodes (20 per combination). `--episodes N` (multiple of 6) runs a balanced prefix for smoke runs; the verdict is only valid at the full 120 and the report marks shorter runs "not a gate run".
- **Runtime:** ≈ 23 s per episode for the default scheme (lockstep-batched, MPS) → ≈ 46 min per Policy gate run; expert seconds.
- **Precision at 120** (95 % Wilson on a rate): observed 0 % → 0–3.1 %; 5 % → 2.3–10.5 %; 25 % → 18.1–33.4 %. (60: 0–6.0 / 1.7–13.7 / 15.8–37.2; 240: 0–1.6 / 2.9–8.5 / 19.9–30.8 at ≈ 92 min.) A good driver (≤ ≈ 5 % crashes) and `wam-v1` today (≈ 25 %) are cleanly separated; thresholds in [Metrics and gates](05-metrics-and-gates.md) must be resolvable at this count.
