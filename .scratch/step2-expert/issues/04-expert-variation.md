# Expert variation knobs and demonstration noise

Labels: wayfinder:grilling
Status: closed
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
- Q1 actions → per tick the expert yields **`clean`** (the label), **`executed`** (clean + noise, clipped) and **`perturbed: bool`**; Step 3 records all three. The design assumes Step 3's default window rule is a **perturbation-free 16-tick future**, so executed = clean inside a window's target and actions agree exactly with the future frames, while the observed frame/State may sit off-line from an earlier burst (recovery data). Burst ticks are only ever starting context, never action targets. DART's noisy-window/clean-label pairing was rejected because it breaks action–video agreement.
- Q2 noise model → **steer-only triangular bursts** (no throttle noise: the deadbeat throttle cancels it within one tick). Per burst: duration ~ U[0.5, 1.5] s, amplitude `burst_peak × U[0.5, 1]` with a random sign, exponential gaps with mean `burst_gap_s`. Per-episode knobs `burst_peak` (default 0.4; 0 = clean) and `burst_gap_s` (default 3.0). About 50% of ticks can start a clean-future window at the defaults. Ranges belong to Step 3; this ticket records the safe edges.
- Q3 style knobs → only **`a_lat_plan`** (default 5.0 m/s²) and **`v_cap`** (default 10 m/s), **fixed by default**. Unobservable per-episode style makes labels ambiguous for the same State (e.g. brake vs accelerate at 5.5 m/s on an end arc), adding modes that get mixed up with the Fork bimodality Step 10 measures. Step 3 may vary them only as a deliberate experiment, with that caveat in the spec. Gains (ωs) and the 20° alignment rule are constants; there is no lateral-offset knob. Safe edges for both knobs are still swept here.
- Q4 seeding → the expert takes the **same `seed` as `reset`**, with two independent streams: **coin = `default_rng([seed, 1])`**, **noise = `default_rng([seed, 2])`** (disjoint from the env's `default_rng(seed)`). Changing noise knobs or turning noise off never changes the `any` branch sequence. Fixed draw order: coin at reset if needed, then one per section exit; noise: gap, duration, amplitude, repeat. With `burst_peak = 0` the noise stream is not drawn. An episode is reproducible from (seed, StartSpec, Instruction, knobs).
- Q5 coin weight → episode knob **`p_outer`** (default 0.5, range [0, 1], used only under `any`): outer iff `coin_rng.random() < p_outer`, one draw per flip. It changes only mode weights, not hidden style, and enables a "does flow reproduce 70/30?" experiment.
- **Edge sweep (prototype `edges`, plus targeted runs, all with bursts unless noted, 1000 seeds × 6 Instruction × Direction = 6000 episodes per variant).** The first pass exposed two rule changes:
  1. **Burst grace period:** every remaining crash at the defaults was a burst within about 10 ticks of reset, stacked on a ±1 m / ±10° start pose. Rule: **no bursts for the first 2 s** (the start pose already provides the recovery example).
  2. **Start-branch rule replaced by a feasibility rollout** (supersedes ticket 03's 20° rule). Seed 838 (`any`, cw) started 0.9 m past the Fork, 0.8 m to the inner side and 15° toward inner, while the coin said outer. That is under 20° but unrecoverable, and it crashed with no noise. New rule at every reset: draw the preferred branch (the Instruction, or one coin under `any`), then **simulate the clean controller on it for 3 s with `step_state`**. Keep it unless the rollout Crashes or is ever (including the start state) > 2.5 m from that branch's Route; then take the other branch if it passes, else keep preferred. The safety-rule switches under `any` dropped from 85 to 0.
  - Final results (crashes / 6000): clean 0; defaults 0; gap 1.0 s 0; a_lat 4.0 + gap 1.0 0; a_lat 5.5 + gap 3.0 0; **a_lat 5.5 + gap 1.0: 2**; v_cap 6 + gap 1.0 0; p_outer 0.7 0 (67% outer). Earlier pass: peak 0.5 → 3–4, peak 0.6 → 64, 0.8 → 1181; a_lat 6.0 → 3 at default gaps. Safety-rule switches: 0 in every variant.
  - Base sweep with the new rules (clean, 1000 seeds × 6): 0 crashes, 0 wrong passages, `any` 49.7% outer. At the defaults, 24% of ticks are perturbed.

**Resolution (2026-11-03):**

| Knob | Meaning | Default | Safe range (0 crashes / 6000) | RNG |
|---|---|---|---|---|
| `burst_peak` | max steer amplitude of a burst; per burst `peak·U[0.5,1]·±1`; 0 = clean | 0.4 | [0, 0.4] | noise `[seed, 2]` |
| `burst_gap_s` | mean of the exponential gap between bursts | 3.0 s | ≥ 1.0 s | noise `[seed, 2]` |
| `a_lat_plan` | lateral accel used to plan curve speeds | 5.0 m/s² | [4.0, 5.0] (5.5 only with gaps ≥ 3 s) | none (fixed per episode) |
| `v_cap` | speed-profile cap | 10 m/s | [6, 10] | none |
| `p_outer` | P(outer) per coin under `any` | 0.5 | [0, 1] | coin `[seed, 1]` |

- **Outputs per tick:** `clean`, `executed`, `perturbed`. Step 3 targets are clean-future windows only.
- **Bursts:** steer only, triangular, duration U[0.5, 1.5] s, exponential gaps, **none in the first 2 s**. Noise draw order: gap, duration, amplitude magnitude, sign, repeat. Nothing is drawn when `burst_peak = 0`.
- **Style knobs are fixed by default**, because they are hidden variables (caveat for Step 3). Gains ωs = 0.5, the 3 s rollout, 2.5 m off-route threshold and 2 s grace are constants.
- **Seeding:** the expert gets the reset `seed`; coin `default_rng([seed, 1])`, noise `default_rng([seed, 2])`. Coin draw order: one draw at reset (always), one per fork-section exit.
- **Start branch:** feasibility rollout (above), replacing ticket 03's 20° heading rule.
