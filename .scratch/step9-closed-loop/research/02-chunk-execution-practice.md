# How chunked policies are executed in practice

Research for ticket [02](../issues/02-chunk-execution-practice.md). Question: how much of a predicted Chunk do action-chunking policies execute, how do they replan, and what goes wrong at multi-modal decisions (our Fork under `any`)? Constraint for us: no retraining of `wam-v1`.

Confidence marks: **[verified]** read in the source during this research; **[secondary]** from a summary page, abstract or docs, not the paper body; **[unsure]** my recollection or inference, not checked.

## 1. Findings by source

### Diffusion Policy (Chi et al., RSS 2023) — receding horizon
<https://arxiv.org/abs/2303.04137> (HTML v5)

- The policy predicts T_p actions and executes T_a of them before replanning (receding-horizon control). The common setting is T_p = 16 and T_a = 8 at 10 Hz. The paper's ablation found **8 steps best for most tasks**. A longer T_a gives more consistency, and "too long a horizon reduces performance due to slow reaction time". **[verified]** (T_p = 16 is confirmed in the BID paper's setup; LeRobot's `diffusion` defaults are horizon 16, `n_action_steps` 8 **[unsure]**.)
- On multimodality: with single-step policies, "consecutive actions could be drawn from different modes, resulting in jittery actions". Sequence prediction lets Diffusion Policy "commit to only one mode within each rollout" (the Push-T left/right example). **[verified]** That commitment comes from long, smooth Chunks, not from any cross-Chunk trick.
- The paper says receding horizon "allows … warm-starting the next inference setup with previous action sequence prediction". **[verified]** Whether the released code actually does this is **[unsure]**. I believe it samples fresh noise each time.
- Sampling: DDIM with 10 inference steps, about 0.1 s on a 3080. **[verified]**

### ACT (Zhao et al., RSS 2023) — temporal ensembling
<https://arxiv.org/abs/2304.13705> (HTML v1)

- Chunk k = 100 at 50 Hz. Without ensembling, success rose from 1 % at k = 1 to 44 % at k = 100, then dipped at k = 200/400 (reactivity). **[verified]**
- **Temporal ensembling:** the policy is queried at every Tick, and all overlapping predictions for the current Tick are averaged with weights w_i = exp(−m·i). w_0 is the **oldest** prediction, which gets the largest weight. A smaller m lets new observations in faster. The paper gives no m value in the text (the code default is m = 0.01 **[unsure]**). The gain was modest: +3.3 % for ACT. **[verified]**
- Cost: one inference per Tick (execute-1).

### π0 (Black et al., Physical Intelligence 2024)
<https://arxiv.org/abs/2410.24164> (HTML)

- H = 50. On 20 Hz robots it runs inference every 0.8 s, after **16 of 50** actions (32 %). On 50 Hz robots it runs every 0.5 s, after **25 of 50** (50 %). **[verified]**
- "We tried temporal ensembling early on and found that it hurt policy performance". Instead it executes action chunks **open loop**, with no aggregation. **[verified]**
- 10 Euler flow steps; about 73 ms on-board (27 ms of that is the 10 flow steps). **[verified]** This is the same sampler shape as `wam-v1` (10 Euler steps, about 66 ms on MPS).
- π0.5 / π0.6 inherit chunked execution. Later PI work adds real-time chunking, below. A separate π0.5 paper rule for execute-k was not checked **[unsure]**. LIBERO evals of π0 commonly execute the first 5 actions **[secondary]** (<https://arxiv.org/pdf/2511.19433v1>).

### Bidirectional Decoding, BID (Liu et al., 2024) — sample selection across replans
<https://arxiv.org/abs/2408.17355> (HTML)

- Draws N samples per step (N = 16 in the main experiments) and picks one using two criteria. **[verified]**
  - **Backward coherence:** L_B = Σ_τ ρ^τ ‖a^(t)_{t+τ} − a^(t−1)_{t+τ}‖₂ over the overlap with the previously **selected** Chunk. ρ decays the weight for further-out steps. It "favors samples that are close to the sequence selected at the previous step."
  - **Forward contrast:** pull towards samples from a strong checkpoint, push away from samples of a weak, early checkpoint (K = 3 nearest per set). It needs a second, under-trained checkpoint, and ablations show both terms matter.
- It makes "no changes to the learned policy". **[verified]**
- Evaluated with action horizon 1, i.e. full closed loop, replanning every Tick. **[verified]**
- Mode switching is reported directly. Vanilla closed-loop sampling showed "oscillations between different latent strategies" and "jittery behavior". Receding horizon jitters when demonstrations are multimodal (left- vs right-handed). **[verified]**
- Baselines: EMA blending of overlapping Chunks (λ = 0.5) weakens commitment ("naive averaging") and is sensitive to its rate. **Warmstart**, which initialises the diffusion noise from the previous decision, degraded 3 of 7 tasks. BID gave about +32 % relative over vanilla on 7 sim tasks, and about +46 % combined with EMA. **[verified]**
- "There is no universally optimal choice of action horizon": long horizons win in deterministic settings, short ones under noise. **[verified]**
- Cost: 13.2 ms → 25.2 ms (8 samples) for VQ-BeT on an A5000, because the samples run in parallel. **[verified]**

### Real-Time Chunking, RTC (Black, Galliker, Levine; Physical Intelligence 2025)
<https://arxiv.org/abs/2506.07339> (HTML v2)

- Purpose: **asynchronous** execution under inference delay d. The next Chunk is generated while the current one runs. **[verified]**
- Method:
  - The first d actions are frozen to the already-committed ones, and the rest of the overlap is **inpainted**.
  - The inpainting uses ΠGDM-style guidance on the flow velocity with a **soft mask**: weight 1 on the first d, exponentially decaying across the overlap, 0 on the last s. The execution horizon satisfies d ≤ s ≤ H − d.
  - The guidance weight is clipped at β = 5, which is needed for stability with few denoising steps.
  - Real robot: π0.5 with 5 steps. **[verified]**
- Training-free, but needs a vector-Jacobian product through the model at each step: 97 ms vs 76 ms for vanilla π0.5. **[verified]**
- Mode switching is named explicitly: "adjacent chunks may jump between different modes (or 'strategies')". Without inpainting the transition "may be arbitrarily discontinuous and out-of-distribution". With **small d the guidance is weak**, "which can allow for the new chunk to still switch strategies". **[verified]**
- Baselines:
  - Temporal ensembling "performs poorly across the board, even with an inference delay of d=0", because "averages of valid actions are not necessarily valid". On the real robot it oscillated enough to trigger protective stops.
  - BID was beaten by RTC in sim and costs more (223 ms with π0.5).
  - **[verified]**
- LeRobot ships RTC for π0, π0.5 and SmolVLA: `execution_horizon` 8–12, `max_guidance_weight` 10 for 10-step flow, an EXP/LINEAR/ONES/ZEROS prefix schedule. **[secondary]** <https://huggingface.co/docs/lerobot/main/en/rtc>

### Training-time RTC (Black, Ren, Equi, Levine; PI, Dec 2025)
<https://arxiv.org/abs/2512.05964>

- Simulates the delay in training and conditions on the clean action prefix, which removes the guidance backward pass. Used with π0.6. **Needs retraining**, so it is out of scope for us. **[secondary]**

### Newer test-time tricks (2026), relevant but not read in full

- **PAINT** (Ho et al., 2026) <https://arxiv.org/abs/2606.19774>: picks the **initial noise** so that the *unmodified* flow ODE reproduces the executed prefix. It uses backward-Euler inversion and needs "no gradients, retraining, or policy modification". It reports performance comparable to RTC with better prefix consistency, and notes that OT flow matching's noise→output map is approximately local. **[secondary, abstract + search summary]**
- **DVAC** (2026) <https://arxiv.org/abs/2606.03847>: chooses execute-k adaptively from the variance of the clean-action estimates over the last denoising steps. It executes the stable prefix and replans before high-variance steps. π0.5 on LIBERO: 94.75 → 98.00 %, with 43 % fewer replans. Test-time only. **[secondary]**
- **TempoWAM** (Ye et al., 2026) <https://arxiv.org/abs/2608.09492>: adaptive execution for *world action models*. A progress monitor decides whether to keep executing or replan. Its monitor and calibration seem to need extra components, so whether it is training-free is **[unsure]**.
- Warm-start / action-reuse families (RTI-DP, Streaming DP, RNR-DP, A2A, …) mostly change training or the noise schedule. **[secondary]**, via <https://arxiv.org/pdf/2602.08245v1>.

### FLUX 3 Action reference (`/Users/grue/code/flux-action`, read only)

- `PolicyConfig.n_action_steps` sets the execute-k. `select_action` keeps a queue of `n_action_steps` actions and replans only when it is empty. The policy is synchronous, with plain open-loop execute-k and no ensembling (`src/flux_action/policy.py` around l. 1077–1108).
- The settings per recipe differ:

  | Recipe | Chunk | Executed before replan | Control rate | Sampler |
  |---|---|---|---|---|
  | DROID | 32 | 32 (whole Chunk, open loop) | — | UniPC, 4 steps |
  | SO-101 | 42 | 32 (`docs/setup.md`) | 30 Hz | — |
  | **games** (`configs/games/train.json`) | 32 | **8** (25 %) | **15 Hz** | 1-step |

  The games recipe is the closest analogue to our toy.
- "RTC arguments are unsupported" (`NotImplementedError`). No sample selection, no seed reuse, and no inpainting are present.

## 2. Practices at a glance

| Practice | Who | Execute-k (of chunk H) | Cross-Chunk consistency mechanism | Cost per Tick | Retraining? |
|---|---|---|---|---|---|
| Open-loop chunks | π0 (16/50, 25/50), FLUX DROID (32/32), FLUX games (8/32) | 25–100 % | none: rely on in-Chunk commitment + long k | 1 inference / k Ticks | no |
| Receding horizon | Diffusion Policy | 8/16 (50 %) | none (paper mentions warm-start) | 1 / k | no |
| Temporal ensembling | ACT | 1 (query every Tick) | weighted average of overlapping predictions, oldest heaviest | 1 / Tick | no |
| EMA blending | BID baseline | 1 | average new with previous (λ = 0.5) | 1 / Tick | no |
| Warm-start noise from previous Chunk | DP (stated), BID baseline | any | initial noise derived from previous decision | 1 / k | no (but hurt 3/7 in BID) |
| Bidirectional decoding | Liu et al. 2024 | 1 (any works) | best-of-N by backward coherence to the previous selected Chunk (+ forward contrast) | N samples / replan, batched | no (needs a weak checkpoint for forward contrast) |
| RTC (inference-time) | PI 2025; LeRobot | d ≤ s ≤ H − d | soft-masked guided inpainting of the overlap | ≈ 1.3× (VJP) | no |
| RTC (training-time) | PI Dec 2025 | same | trained prefix conditioning | 1× | **yes** |
| Initial-noise selection (PAINT) | 2026 | async | invert the ODE to find noise consistent with the prefix | inversion + 1 sample | no |
| Adaptive k (DVAC, TempoWAM) | 2026 | variable | replan when uncertainty is high / progress stalls | variable | no / unclear |

Consensus:
1. Executing 25–50 % of a Chunk before replanning is the norm. Whole-Chunk open loop is used when inference is expensive.
2. **Averaging Chunks (ensembling/EMA) is the documented failure at multimodal decisions.** It produces the in-between action, which here means steering into the Island.
3. Cross-Chunk mode switching is a recognised problem. The no-retraining fixes are selection against the previous Chunk (BID) and inpainting or noise choice anchored on the previous Chunk (RTC, PAINT).

## 3. Fit for our toy

Facts that matter:
- Chunk is 16 Ticks, about 1.07 s at 15 Hz.
- Replan costs about 70 ms at B = 1 and 21.5 ms per Chunk batched on MPS.
- **The sim waits for the model, so inference delay d = 0.** That removes RTC's main reason to exist (latency) and leaves only its consistency effect.
- `wam-v1` commits cleanly inside one Chunk (commit 0.99). Under `any`, though, each replan is a fresh coin flip until the car's position settles the branch.
- The flip-flop window is the approach to the Fork. With k = 8 there are maybe 2–4 replans there **[unsure; 01 / 03 should measure]**.

Wall-clock cost per 900-Tick episode at B = 1 (estimates):

| k | Replans | Time |
|---|---|---|
| 16 | 57 | ≈ 4 s |
| 8 | 113 | ≈ 8 s |
| 4 | 225 | ≈ 16 s |
| 1 | 900 | ≈ 63 s |

Lockstep batching of episodes cuts these by about 3×. Every option below is affordable. Cost decides batching, not the scheme.

What ticket 03 should prototype, in priority order:

1. **Execute-k sweep, plain receding horizon, first sample:** k ∈ {16, 8, 4}, with an optional k = 2 as the reactivity end. This is the baseline that literature and FLUX (games: 8/32 at 15 Hz) use.
   - Expect: k = 8 as the default, k = 16 to be weak on recovery (off-Route), and small k to be more reactive but to flip more at the Fork.
2. **Shared noise across replans (seed fixed per episode):** call `act(..., seed=episode_seed)` with the *same* integer every replan instead of seeding from episode + Tick.
   - In `wam-v1` this gives identical initial `x_z`, `x_a` (`flow.initial_noise`). The ODE map from observation to action is continuous, so neighbouring Frames on the approach should land on the same branch.
   - Zero cost, zero code beyond the seed rule. It is the training-free cousin of PAINT's "pick the noise, keep the ODE".
   - Different episodes still get different branches, so the `any` balance is preserved across episodes.
   - Risk: a correlated noise draw could bias the whole episode. Compare against per-replan fresh noise. **[my proposal; the literature's warm-start variant hurt in BID, but that is a different trick (noise from the previous action) — unsure how fixed noise fares]**
3. **BID-lite selection (backward coherence only):** n_samples = N (8 is a good start, batched in one `act` call). Pick the sample that minimises Σ_τ ρ^τ ‖a_new[τ] − a_prev[k+τ]‖ over the 16 − k overlapping steps of the previously *selected* Chunk, with ρ ≈ 0.9.
   - Drop forward contrast: it needs a weak checkpoint, which we don't have. A Step 5–8 early Run checkpoint could serve later, if one exists.
   - Under `any` this keeps the branch already chosen. Under `outer`/`inner` it is harmless.
   - The first replan of an episode takes sample 0.
   - Cost: ≈ 21.5 ms × N/B-scaling. **[unsure, measure]**
4. **Negative control: temporal ensembling (ACT, m = 0.01, execute-1) or EMA.** This is literature-predicted to fail at the Fork by averaging branches.
   - Optional and cheap. It would show the failure on our own Fork, which is useful for the write-up but not needed for the default.
5. **Hard-prefix inpainting, deferred:** RTC-like, without guidance. At each Euler step, overwrite the overlapping part of `x_a` with the previous Chunk's remaining actions re-noised to the current t (check the `flow.noised` convention), with a decaying soft mask.
   - Training-free. It needs a custom Euler loop around `policy.model` in `prototypes/` (`flow.sample` has no prefix hook).
   - Try it only if 2 and 3 don't stop flip-flops. Full RTC guidance needs autograd through the model (≈ +30 % time) and is weaker at d = 0 anyway.

Don'ts:
- Don't average samples or Chunks, not across samples (n_samples > 1) and not across time.
- Don't adopt training-time RTC or warm-start families, since they need retraining.

Other knobs for 03 (from the sources):
- π0 and our sampler agree on 10 Euler steps. Keep `n_steps = 10` unless 01 shows slack.
- CFG is not used on actions in FLUX DROID (`guidance_scale_action` 1.0, video 4.0). Keep `cfg_action = cfg_video = 1` as the default and leave CFG to Step 10.
- Adaptive k (DVAC-style, from variance across the final denoising steps or across samples) is a possible later remedy, not a default.

Parameters the driver should expose, so 03 can sweep without code changes:
- `execute_k`
- `n_samples`
- `select` (`first` | `coherence`), with `rho`
- `seed_mode` (`per_replan` | `per_episode`)
- `n_steps`, `cfg_action`, `cfg_video`
