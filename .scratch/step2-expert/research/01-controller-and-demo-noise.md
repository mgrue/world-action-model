# Research: path-tracking controller, speed profile and demonstration noise

Ticket: ../issues/01-controller-research.md. Researched 2026-11-03. I read the literature (sources at the end) and ran throwaway experiments against the real `toycar` simulator (scripts in a session scratch dir, not in the repo).

## Recommendation

### 1. Controller: curvature feed-forward plus lateral and heading feedback

Use the rear-wheel-position feedback law (Samson; Paden et al. 2016 §V-A). PLAN.md says "pure pursuit". Treat that as a placeholder: this law is better on every axis that matters here.

The law is a curvature command:

```
k_cmd = κ̄·cos θe / (1 − κ̄·e)  −  kθ·θe  −  ke·e·sinc(θe)        (sinc(x) = sin x / x)
steer = atan(L·k_cmd) / 35°        after clamping |L·k_cmd| ≤ min(tan 35°, 8·L / v1²)
```

The terms, all in the travel frame:

- **e:** the signed lateral offset from the followed Route, positive to the right of the travel direction. It is `d·Projection.lateral_m`, with d = +1 for cw and −1 for ccw.
- **θe:** `wrap(θ − (tangent + π·[ccw]))`.
- **κ̄:** the travel-signed *mean* path curvature over the next tick's arc length, `wrap(θ_path(s + d·v1·DT) − θ_path(s)) / (v1·DT)`. Use this, not the point value `curvature_at(s)`. It handles the curvature sign flips in the S-bends exactly. In testing it cut steady-state RMS error from 2.5 cm to 1.4 mm.
- **v1:** the post-throttle speed, `clip(v + a·DT, 0, 10)`. The simulator clamps the steer using v1, so the expert must predict it.

**Why this law fits the simulator.** The State reference point (the rectangle centre) evolves with the rear-axle-form equations: θ̇ = v·k and k = tan δ / L. The point therefore follows a circle of exactly the commanded curvature. Steering is also instant. Together these make the feed-forward exact: on the centerline with θe = 0, the car tracks the line-and-arc path with zero error, and the feedback only removes start-pose and noise offsets.

**Why not the alternatives.** Pure pursuit and Stanley are feedback-only on this path:

- Pure pursuit cuts corners and has a steady-state offset on arcs. Larger lookahead means more cutting.
- Stanley assumes front-axle kinematics, which this simulator does not have, and its gains are speed-coupled.

**Starting gains: one spatial bandwidth ωs = 0.5 m⁻¹, critically damped.** That gives ke = ωs² = 0.25 m⁻² and kθ = 2ωs = 1.0 m⁻¹. Results were insensitive across ωs = 0.25 to 1.0. Linearised, the law gives e'' + kθ e' + ke e = 0 in *distance*, so convergence is speed-independent: about 4 m to settle. Clamp `1 − κ̄e ≥ 0.2`.

**Route choice.** Choose the followed Route by the Instruction, never by nearest projection. Near the Fork and Merge the branches are close together but their headings differ by up to about 67°. My first harness picked the nearest Route, and that alone caused the only crashes in clean runs: seeds 3 and 129, both started on a branch near the Merge.

### 2. Speed profile: forward and backward pass over station

Precompute the profile once per (Route, Direction) on a 0.05 m station grid in travel order:

1. Set `v_lim(s) = min(v_cap, sqrt(a_lat_plan / |κ(s)|))`.
2. Forward pass: `v[i] ≤ sqrt(v[i−1]² + 2·a_acc·ds)`.
3. Backward pass: `v[i] ≤ sqrt(v[i+1]² + 2·a_brk·ds)`.
4. Run both passes twice around the closed loop so that wrap-around is consistent.

**Throttle.** Set `vref = profile(s + d·(v·DT + 0.5 m))`, then use a deadbeat throttle: `(vref − v)/DT` divided by ACCEL or by BRAKE. That is exact for this physics.

**Starting parameters:**

- **a_lat_plan = 5.0 m/s²**, which is 62% of the 8 m/s² clamp. That gives 4.94 m/s on the branch arcs and 6.12 m/s on the end arcs. This margin matters more than any steering gain, because the clamp also limits the *feedback's* authority (see the empirical results).
- **a_brk = 6 m/s²**, with 2 m/s² in reserve under the 8 m/s² limit.
- **a_acc = 4 m/s²**.
- **v_cap = 10 m/s**.
- At a_lat_plan = 5, the expert averages about 6 m/s, roughly 4.1 laps per 900-tick episode.
- Expose a_lat_plan and v_cap as expert-style knobs, for example a "cautious" range of 4 to 5.5 m/s².

### 3. Demonstration noise: inject on the executed action, record a clean label, shape the noise as temporally correlated bursts, and keep randomised starts

1. **Noise is needed.** With clean execution, the expert converges to the centerline within about 4 m and then holds it with 1 mm RMS error. Off-nominal states would then appear only in the first second of each episode. DART (Laskey et al. 2017, Prop. 4.1) makes the formal point: a deterministic supervisor never shows the states a slightly-wrong learner will reach. Codevilla et al. 2018 are the closest analogue (driving, with triangular steering-noise bursts on about 10% of the data). Removing their noise-injected data cut success from 88% to 56% in Town 1 and from 64% to 22% in Town 2. On the physical truck, interventions rose from 0.67 to 8.67.

2. **Execute noisy, record clean.** Execute `a_exec = a_expert + n` and record `a_expert` as the training label. Record `a_exec` as well, because the frames are caused by `a_exec`. This is DART's pairing, and Codevilla also trains only on the driver's corrective signal. Recording the noisy action as the label would be wrong for a flow-matching policy in particular: flow matching models the whole action distribution, so it would learn to *reproduce* the noise.

3. **Shape the noise for 16-step chunks and the frame stream.** PLAN.md Step 6 trains frames and actions jointly, and each window holds 16 future frames and 16 future actions. If noise is injected inside the 16-tick future of a window, the future frames show drift that the clean action labels do not explain. The chunk label then also contains the expert's reactions to future noise, which the model cannot predict from the observed frame. So:
   - Use **sparse, temporally correlated bursts**: a triangular steer offset lasting about 1 s (15 ticks), with random sign and a peak of U[0.5, 1]·0.4 steer units. Leave exponential gaps averaging about 3 s between bursts, which gives roughly a 25% duty cycle.
   - Write a per-tick `perturbed` flag so that Step 3 can cut training windows whose 16 future ticks are noise-free. In those windows `a_exec = a_expert`, the frames and actions agree exactly, and the window shows a genuine expert *recovery* from the perturbed state.
   - If every window must be usable, a deterministic simulator also allows **counterfactual relabelling**: from the window's start state, roll out the clean expert for 16 ticks and render that rollout. This costs about 17× more simulation. It can stay optional in the spec.
   - Correlation matters because i.i.d. per-tick noise at a safe scale barely moves the car: it averages out within a chunk. Bursts produce the 0.3 to 0.7 m, 5 to 15° excursions that resemble learner drift. The open-loop chunk execution that ACT and Diffusion Policy rely on drifts in exactly that correlated way.

4. **Keep randomised starts** (Step 1 already does this: ±1 m, ±10°, 0 to 6 m/s). ChauffeurNet relies on the same idea, using synthetic start and midpoint perturbations of ±0.5 m and up to ±60° heading. Randomised starts give off-nominal *initial* states for free. Bursts give them mid-lap, including at the Fork.

5. **DAgger is not needed for the expert spec, but keep it possible.** The scripted expert can label any state for free, which makes DAgger (Ross et al. 2011) trivially possible later, in Step 9 or as an ablation. The requirement for the spec is that the expert must be **a pure function of (State, Instruction, small explicit memory such as the chosen Branch)**. It must not keep hidden integrator state, so that it can be queried on states it never visited.

6. **Noise scale.** DART's rule is to match the expected learner error. Before Step 9 that error is unknown, so use the simulator's safety budget instead: no noise-induced crashes in 100 episodes, and mean peak |e| below about 0.6 m. The burst peak (0.4 steer units, about 14° of steer) is the knob. Expose it so the dataset can sample it, and DART-style recalibrate it once Step 9 measures real policy error.

## Empirical results (`toycar` simulator, 100 random-start episodes of 900 ticks)

Controllers ("ff+fb" is the recommended law; "pp" is speed-scaled pure pursuit with Ld = max(2.5, 0.6·v1); "rms" is RMS |e| after the first 4 s):

| Setup | Crashes | Mean peak \|e\| | rms \|e\| |
|---|---|---|---|
| FF only (no feedback), a_lat 6 | 198/200 | 1.32 | 0.675 |
| ff+fb ωs 0.25 / 0.5 / 0.8, a_lat 6 | 0 (after route-choice fix) | 0.53 / 0.51 / 0.50 | 0.002 |
| pure pursuit, fixed Ld 3 / 5 m | 0 | 0.57 / 0.97 | 0.125 / 0.372 |
| pure pursuit, Ld = max(2.5, 0.6 v) / max(3, 1.0 v) | 0 | 0.59 / 1.17 | 0.177 / 0.546 |
| Stanley (front point at +L/2, k = 2.5, k_soft = 1) | 0 | 0.63 | 0.311 |
| ff+fb, point curvature instead of tick-mean curvature | 0 | 0.55 | 0.025 (vs 0.0014) |

Peak |e| is dominated by the start offset. Speed-scaled pure pursuit cuts corners more as v rises, as expected.

Noise robustness (executed steer noise in steer units):

| Noise | ff+fb, a_lat 6 | ff+fb, a_lat 5 | pp, a_lat 5 |
|---|---|---|---|
| none | 0 crashes | 0 | 0 |
| i.i.d. σ 0.2 / 0.3 | 0 / 14 | 0 / 0 (rms 0.05 / 0.11) | 0 / 0 (rms 0.18 / 0.23) |
| OU σ 0.1 / 0.15 / 0.2, τ 0.5 s | 3 / – / 85 | 0 / 5 / – | 0 / 19 / – |
| bursts, peak 0.3 / 0.5 / 0.6, 1 s | – / – / 32 | 0 / 0 / 0 (mean peak \|e\| 0.52 / 0.58 / 0.70) | 0 / 0 / 12 |

What these results show:

- **Lateral-acceleration headroom decides robustness.** At a_lat_plan = 6, which is 75% of the clamp, the feedback can add only about 0.045 to 0.07 m⁻¹ of curvature on the arcs, so outward noise cannot be corrected. Most crashes were on the outer end arcs at 6.7 m/s. Dropping to 5 m/s² removed nearly all of them.
- **The crash budget is about 1 m of lateral error, not 1.55 m.** With 10 to 20° of heading error, a front corner leaves the 2.8 m band at e ≈ 0.9 to 1.2 m.
- **Sustained OU noise is far harsher than bursts at similar RMS.** I would not use continuous OU noise.
- **Simulation cost.** Simulation plus two projections per tick costs roughly 0.1 ms in pure Python. 100 episodes take about 8 s per configuration.

## Findings per topic

### Pure pursuit (Coulter 1992; Paden et al. 2016; Snider 2009)

- **Law:** fit an arc from the reference point to a goal point at lookahead distance Ld on the path: `k = 2 sin α / Ld`.
- **Strengths:** robust and simple. It is stable for straight paths, and it is undefined when the car is more than Ld from the path.
- **Weaknesses:** it shows steady-state error on constant-curvature arcs and corner cutting when curvature changes, and it is sensitive to Ld. Small Ld oscillates; large Ld cuts corners.
- **Speed scaling:** the usual fix for high-speed sensitivity is `Ld = k·v`. It trades the sensitivity for more cutting.
- **Snider's conclusion:** no single method dominates; pure pursuit is robust, and Stanley tracks better at moderate speed on smooth paths.

### Stanley / front-wheel feedback (Hoffmann et al. 2007; Paden et al. 2016)

- **Law:** `δ = θe + atan(k·e_f / (k_s + v))` on the *front-axle* crosstrack error e_f.
- **Stability:** locally exponentially stable for C¹ paths, but not in reverse.
- **Practical additions:** implementations usually add yaw-rate damping and a steering-delay term.
- **Fit here:** it is derived for front-axle kinematics, which do not match this simulator's rear-axle-form point. It also has no curvature feed-forward, so it lags on arcs. This showed up as 0.31 m RMS.

### Rear-wheel feedback with curvature feed-forward (Samson; Paden et al. 2016 §V-A)

- **Law:** `ω = v κ cosθe/(1 − κe) − g1 θe − k2 v e·sinθe/θe`, with g1 = kθ|v|. That gives speed-independent spatial convergence.
- **Fit here:** this is exactly the "feed-forward from curvature at the projected station plus PD on lateral and heading error" family. It is the natural law when the controlled point obeys θ̇ = v·k, which is this simulator's case.
- **Caveat:** the formal guarantee assumes a C² path, and this track's curvature is piecewise constant. With instant steering, the tick-mean curvature κ̄ restores exact tracking across the jumps, as shown empirically above.

### Speed profiles (racing and velocity-planning literature)

- **Standard approach:** take the lateral limit `v = sqrt(a_y/|κ|)`, run a forward pass limited by acceleration and a backward pass limited by braking, then take the pointwise minimum. Examples are the UNIST velocity planner and arXiv 1902.00606.
- **Why it is simple here:**
  - curvature is piecewise constant, so the profile is a few plateaus joined by √-shaped ramps;
  - the backward pass makes braking finish *at* the arc entry;
  - the forward pass caps exit acceleration.
- **Margins:** the literature applies margins by using a fraction of the friction or GG limit for robustness. I applied the same idea to the 8 m/s² clamp.

### DART (Laskey et al. 2017)

- **Method:** collect off-policy with noise N(0, Σ) added to the supervisor's *executed* control, and train on the supervisor's noise-free action at each visited state.
- **Choosing Σ:** iterate it to match the trained robot's error, scaled to a target trace α.
- **Results:**
  - DART matched DAgger on MuJoCo, about 3× faster on Humanoid, without the cost of letting a bad policy drive.
  - On the grasping task it reached 79% success against 49% for behaviour cloning (α = 3).
- **Noise shape:** per-step Gaussian. The paper does not study temporal correlation.

### DAgger (Ross, Gordon, Bagnell 2011)

- **Method:** iterate. Roll out the learner, have the expert label the states it visits, aggregate the data and retrain. This gives no-regret guarantees on the learner's own state distribution.
- **Cost:** it needs an expert that can be queried at arbitrary states, plus training rounds inside data collection.
- **Fit here:** with a scripted expert both are cheap, but the learner exists only from Step 7 onward. It is a natural Step 9 ablation, not a Step 2 or Step 3 requirement.

### Correlated perturbations in driving IL (Codevilla et al. 2018; ChauffeurNet, Bansal et al. 2018)

- **Codevilla et al.:** inject *temporally correlated triangular* steering noise (the drift, then the driver's correction, over about 2.5 s) on about 10% of the data, and train only on the driver's signal. That is the source of the 88% vs 56% result in recommendation 3.
- **ChauffeurNet:** perturbs trajectory midpoints by ±0.5 m and up to ±π/3 heading, smooths back to the original, down-weights perturbed examples 1:10, and finds that without perturbations the model "could not recover in any scenario".

### Action chunks (ACT, Zhao et al. 2023; Diffusion Policy, Chi et al. 2023)

- **ACT:** chunking reduces the effective horizon by a factor of k, which reduces compounding error. Success rose from 1% at k = 1 to 44% at k = 100. Chunking also absorbs temporally correlated confounders such as pauses, provided they fall within a chunk.
- **Diffusion Policy:** defaults are Tp = 16 predicted and Ta = 8 executed. Sequence prediction avoids switching modes between steps, and too long a horizon "reduces performance due to slow reaction time".
- **Implication for noise:** inside an executed chunk the policy is open loop, so its errors are correlated over up to Ta ticks. Training data should therefore show recoveries from *correlated* drift. The chunk label must also stay predictable from the window's conditioning, which is why I recommend noise-free 16-tick futures or counterfactual relabelling.
- **Gap:** I found no paper that studies OU or burst noise specifically for chunked IL data collection. This recommendation is inferred from the sources above plus the experiments.

## Sources

- Snider, *Automatic Steering Methods for Autonomous Automobile Path Tracking*, CMU-RI-TR-09-08 (2009): https://www.ri.cmu.edu/publications/automatic-steering-methods-for-autonomous-automobile-path-tracking (the PDF returned 404 when fetched; summary taken from the abstract and secondary sources)
- Paden, Čáp, Yong, Yershov, Frazzoli, *A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles* (2016), §V-A: https://arxiv.org/abs/1604.07446 (read via https://ar5iv.arxiv.org/html/1604.07446)
- Hoffmann, Tomlin, Montemerlo, Thrun, *Autonomous Automobile Trajectory Tracking for Off-Road Driving* (ACC 2007), via https://av1tenth-docs.readthedocs.io/en/latest/assignments/projects/project2.html and https://arxiv.org/pdf/2011.08729
- Thrun et al., *Stanley: The Robot that Won the DARPA Grand Challenge* (JFR 2006): https://ri.cmu.edu/publications/stanley-the-robot-that-won-the-darpa-grand-challenge
- PythonRobotics path tracking (pure pursuit, Stanley, rear-wheel feedback reference implementations): https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/path_tracking.html
- Velocity planning, forward and backward passes: https://hmcl-unist.github.io/unicorn-racing/posts/velocity-planner-en/, https://arxiv.org/pdf/1902.00606, https://arxiv.org/html/2505.05157v1
- Laskey, Lee, Fox, Dragan, Goldberg, *DART: Noise Injection for Robust Imitation Learning* (CoRL 2017): https://arxiv.org/abs/1703.09327 (read via https://ar5iv.arxiv.org/html/1703.09327), https://proceedings.mlr.press/v78/laskey17a.html
- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011): https://arxiv.org/abs/1011.0686, https://proceedings.mlr.press/v15/ross11a
- Codevilla, Müller, López, Koltun, Dosovitskiy, *End-to-end Driving via Conditional Imitation Learning* (ICRA 2018): https://arxiv.org/abs/1710.02410 (read via https://ar5iv.arxiv.org/html/1710.02410)
- Bansal, Krizhevsky, Ogale, *ChauffeurNet* (2018): https://arxiv.org/abs/1812.03079 (read via https://ar5iv.arxiv.org/html/1812.03079)
- Zhao, Kumar, Levine, Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, RSS 2023): https://arxiv.org/abs/2304.13705
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (RSS 2023): https://arxiv.org/abs/2303.04137
- Swamy, Choudhury, Bagnell, Wu, *Causal Imitation Learning under Temporally Correlated Noise* (ICML 2022): https://www.ri.cmu.edu/app/uploads/2022/08/Causal_IL_under_TCN-2.pdf (the inverse problem: noise in the *expert's* data breaks behaviour cloning, which is one more reason to keep labels clean)
- Empirical: throwaway scripts in this session's scratch dir (`ctrl.py`, `noise.py`, `sweep.py`), run with `uv run python` against `src/toycar` at commit bbca2d9.
