# JEPA recipes at small scale

Ticket: [02-jepa-recipes-at-small-scale](../issues/02-jepa-recipes-at-small-scale.md). Desk research, November 2026: the I-JEPA paper and its official configs and code (read directly), V-JEPA / V-JEPA 2, C-JEPA, LeJEPA, collapse-metric papers (RankMe, LiDAR), and two community small-scale runs. The mask numbers for an 8 × 8 grid come from a short simulation of the official sampler logic (scratch script, not committed). Anything marked **guess** is my extrapolation, not something a source measured.

## Summary

- **The official recipe translates to our setup almost unchanged.** I-JEPA uses 4 target blocks (scale 0.15–0.2, aspect 0.75–1.5) and 1 context block (scale 0.85–1.0, square) with the target blocks removed from it. Block sizes are drawn once per batch, positions per image. On an 8 × 8 grid that means 3 × 3 to 4 × 3 target blocks, about 45 predicted tokens and about 21 context tokens per Frame.
- **Target side:** EMA momentum 0.996 → 1.0, linear over all of training. The target is the last layer of the target encoder, LayerNorm'd per token over the feature dimension, with no affine parameters. Targets come from masking the encoder output; the full Frame goes through the target encoder.
- **Predictor:** narrow (width 384 for 768–1280-wide encoders) and fairly deep (depth 6 for ViT-B, 12 for ViT-L/H). A shared learnable mask token plus position is used. Both the narrow width and the extra depth helped in the ablations.
- **Collapse:** the paper adds nothing beyond EMA + stop-grad + predictor asymmetry. C-JEPA and LeJEPA argue that EMA alone is not a guarantee. Small-scale I-JEPA runs work, but they are weak: about 52–56 % CIFAR-10 linear probe with ViT-Tiny after 100 epochs, below VICReg/BYOL on the same backbone.
- **Two surprises for this project:**
  1. **PLAN's "at least half the rectangles cover the car" fights I-JEPA's masking.** Every target is cut out of the context, so a car-covering target hides the car from the context encoder. Then (a) the car cannot be predicted, and (b) the context encoder never sees the car in those samples. JEPA's bias toward *predictable* ("influential") features (Littwin et al.) would then push the car *out* of the features. Random I-JEPA masks already hide the whole car in about 40 % of samples. The car bias should be **partial**: a target block covers some of the car's patches, and the context keeps at least one.
  2. **The Track is fixed, which allows a cheap degenerate solution.** Features that depend only on patch position (plus episode colour) predict every background target perfectly. The resulting loss is near zero, and the per-channel spread across *all tokens* looks healthy. So the "no collapse" check must look at variation **across Frames at a fixed position**, and specifically at car patches, not only at global channel std.

## Masking

**I-JEPA (paper §3 and App. A; [config](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml); [`src/masks/multiblock.py`](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py)):**
- Targets: `num_pred_masks: 4`, `pred_mask_scale: [0.15, 0.2]`, `aspect_ratio: [0.75, 1.5]`. Target blocks may overlap each other.
- Context: `num_enc_masks: 1`, `enc_mask_scale: [0.85, 1.0]`, aspect 1.0. Every target region is removed from it (`allow_overlap: false`). `min_keep: 10` patches.
- Sampling: in the collator, the block **size** (scale and aspect) is drawn once per batch from a step-seeded generator, and **positions** are drawn per image. All masks in a batch are trimmed to the smallest count so that they stack.
- Targets are cut from the target encoder's **output**, not its input. Masking the input instead dropped 1 %-IN linear probe accuracy from 67.3 to 56.1 (Table 11).
- Ablations (ViT-B/16, 300 ep, 1 %-IN linear probe):
  - Target count: 1 → 9.0, 2 → 22.0, 3 → 48.5, 4 → 54.2 (Table 10).
  - Target scale: (0.075–0.2) 19.2; (0.15–0.2) 54.2; (0.2–0.3) 33.6 (Table 8).
  - Context scale: 0.40 → 31.2, 0.85 → 54.2 (Table 9).
  - Multi-block 54.2 vs single block 20.2 vs random patches 17.6 (Table 6).
- In short: several mid-sized targets, a large spatially spread context, and results that are sensitive to all three numbers.

**V-JEPA ([`configs/pretrain/vitl16.yaml`](https://github.com/facebookresearch/jepa/blob/main/configs/pretrain/vitl16.yaml)):** two mask generators per batch. One draws 8 blocks at scale 0.15; the other draws 2 blocks at scale 0.7, aspect 0.75–1.5. The context is the complement. The predictor sees far less context than in I-JEPA, which suggests the image recipe is not the only one that works.

**On our 8 × 8 grid** (simulation of the same sampler, 20 k draws):

| Targets | Block shapes (most common) | Predicted tokens / Frame (sum over blocks) | Context tokens | P(whole car hidden from context) |
|---|---|---|---|---|
| 4 × scale 0.15–0.2 (I-JEPA default) | 4×3 52 %, 3×3 25 %, 3×4 21 % | 45 | 21 | 0.40 |
| 4 × scale 0.10–0.15 | 3×3 57 %, 3×2 27 % | 32 | 27 | 0.27 |
| 2 × scale 0.15–0.2 | as row 1 | 22 | 31 | 0.23 |

The car's 7 × 4 px footprint touches 4 patches in ~73 % of Frames, 2 patches in ~25 % and 1 patch in ~1 % (measured on the rotated bounding box, so this is an upper bound). The grid is coarse: scale steps of 0.05 jump the block area by ~3 patches, so a few discrete block shapes are cleaner than continuous scales.

**Precedent for object-biased masking.** None of it is JEPA. All of it is MIM or contrastive:
- [AttMask](https://arxiv.org/abs/2203.12719) masks the patches the teacher attends to most.
- [SemMAE](https://arxiv.org/abs/2206.10207) masks whole semantic parts, scheduled from easy to hard.
- [ADIOS](https://arxiv.org/abs/2201.13100) learns masks adversarially and reports that ground-truth object masks clearly beat random masks.
- AutoMAE and ObjMAE: object priors in mask sampling help.
- A contrastive-ConvNet paper ([2309.12757](https://arxiv.org/abs/2309.12757)) recommends spreading masks over foreground and background, rather than masking only the object.

The common thread is *hide parts of the object so that the rest of the object must explain them*. None hides the whole object from every view. I found no JEPA paper with object-aware target selection. **Guess:** for JEPA, partial occlusion is the variant that produces a learning signal about the car, for these reasons:
- When the context sees the car body and the target is the patch with the nose, predicting it needs the car's position and heading.
- When the whole car is hidden, the best prediction is the mean car-or-road feature. The encoder then gains nothing from encoding the car, and JEPA's implicit bias toward predictable features ([Littwin et al. 2024](https://arxiv.org/abs/2407.03475)) rewards dropping it.

## Target encoder

From the I-JEPA [`src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py), App. A.1 and the configs:
- **EMA:** `ema: [0.996, 1.0]`, linear in the step over `ipe·epochs` steps (the momentum schedule ignores the LR's 1.25× `ipe_scale` stretch). Target = context encoder at init. The update runs after every optimizer step over all parameters.
- V-JEPA uses `[0.998, 1.0]`. V-JEPA 2 keeps the teacher EMA and weight decay **fixed** instead of ramping them ([2506.09985](https://arxiv.org/abs/2506.09985)).
- At 0.996 the target's half-life is about 170 steps; at 0.999 it is about 690 steps. With only ~20–50 k steps here, the linear ramp to 1.0 freezes the target over the last ~10 % of training, which is the intended annealing.
- **Normalisation:** `h = F.layer_norm(h, (h.size(-1),))`, per token over the feature dimension, without affine parameters. It comes after the ViT's own final LayerNorm. The same line is in V-JEPA.
- **Layer:** the last block's output (after the final norm). No projector head, no `[cls]` token.
- **Evaluation:** uses the **target** encoder (App. A.1: "We use the target-encoder for evaluation").
- **Loss:** the paper says "average L2". The code uses `F.smooth_l1_loss` (β = 1), which for LayerNorm'd targets with errors < 1 equals 0.5 · L2. V-JEPA and V-JEPA 2 use L1 (`loss_exp: 1.0`). So PLAN's squared error matches the paper, and at our scale the choice should matter little (**guess**).

## Predictor

- I-JEPA App. A.1: predictor width fixed at **384**, attention heads equal to the encoder's, depth **6** for ViT-B (12 blocks), **12** for ViT-L/H, **16** for ViT-G. So it is narrower than the encoder (0.3–0.5×) and at least half as deep.
- Ablations:
  - Depth 6 → 12 on ViT-L: 64.0 → 66.9 (Table 12).
  - Width 384 vs 1024 on ViT-L (fine-tune, 1 %-IN): 70.7 vs 68.4 (Table 14). The bottleneck helps.
- Inputs:
  - The context tokens, projected to the predictor width and given their positions.
  - One **shared learnable mask token** per target patch, plus that patch's position.
  - The predictor runs once per target block (M = 4 times), or in one batched pass with the blocks repeated.
  - Its output is projected back to the encoder width; the loss is taken only at target tokens.
  - V-JEPA: `use_mask_tokens: true`, `zero_init_mask_tokens: true`.
- **Positions:**
  - I-JEPA uses fixed 2D sin-cos added to tokens.
  - V-JEPA 2 switches the encoder and predictor to **RoPE** ("helps stabilize training for the largest models").
  - With RoPE, mask tokens need no additive position: they get rotary phases at their grid coordinates in every attention layer, like the context tokens. This is our plan, and it is what V-JEPA 2 does.
  - One caveat (**guess**): with RoPE alone, all mask tokens start identical and differ only through attention to the context. That is fine because they attend to the context in the first layer. If the prototype shows trouble, adding a learned or sin-cos absolute position to the mask tokens only is a cheap fix.

## Collapse

- **What prevents it in I-JEPA:**
  - Architectural asymmetry: an EMA target with stop-gradient, plus a predictor (paper §2 and App. B).
  - No regulariser is used.
- **V-JEPA:** the code contains a VICReg-style hinge on the predictor-output std across patches, `reg_coeff * mean(relu(1 - sqrt(var(z, dim=1) + 1e-4)))`, but it ships with `reg_coeff: 0.0`. So the authors built it and did not need it.
- **C-JEPA** ([2410.19560](https://arxiv.org/abs/2410.19560), NeurIPS 2024):
  - Argues that I-JEPA's EMA "is inadequate in preventing entire collapse" and that the predictor struggles to learn the mean of the patch representations.
  - Adds VICReg on block-averaged context embeddings through a projector: β_vicreg = 0.001, with sim/std/cov = 25/25/1 and γ = 1.
  - Gains are modest at scale (ViT-B 600 ep linear: 72.9 → 73.7) and larger for tiny short runs (ViT-T/16 100 ep: 22.1 → 23.3; ViT-S: 29.6 → 32.4).
  - Its collapse evidence is theoretical, not measured.
- **LeJEPA** ([2511.08544](https://arxiv.org/abs/2511.08544), [code](https://github.com/rbalestr-lab/lejepa)):
  - Drops EMA, stop-gradient and teacher entirely. It adds SIGReg, which pushes embeddings toward an isotropic Gaussian through random 1-D projections and an Epps–Pulley test.
  - Loss = (1−λ)·pred + λ·SIGReg, with λ = 0.05, 1024 slices (code default 256) and 17 integration points over [−5, 5].
  - Two claims matter here: the training loss tracks downstream accuracy (Spearman ≈ 0.85, ≈ 0.99 after rescaling), and small in-domain pretraining beats DINOv2/v3 transfer on Galaxy10.
  - Caveat: LeJEPA is a multi-view *global-embedding* method (2 global + 6 local crops), not patch masking. Using SIGReg on per-patch features of an I-JEPA is an extrapolation (**guess**).
- **How collapse shows up:**
  1. **Complete:** all outputs equal, loss → 0, per-channel std → 0.
  2. **Dimensional:** features span a low-rank subspace. The effective rank falls while the loss looks fine ([RankMe](https://arxiv.org/abs/2210.02885)).
  3. **Here, a third kind (inferred):** features depend only on patch position and episode colour. The car information is dropped, and both the loss and the global std look healthy.
- **Standard diagnostics:**
  - Per-dimension std of embeddings (VICReg-style hinge at 1, or just logged).
  - RankMe effective rank: exp of the entropy of normalised singular values. It has no hyperparameters, but correlates only weakly with I-JEPA quality across mask scales (Kendall τ 0.29).
  - LiDAR ([Thilak et al., ICLR 2024](https://arxiv.org/abs/2312.04000)): rank of the LDA matrix of the SSL task. It needs several masks per image; τ 0.79 for I-JEPA mask-scale sweeps.
  - A periodic linear probe, which is the only direct measure. The JEPA loss itself is *not* a good quality signal for I-JEPA (LiDAR paper).
- **Needed here?** No source shows a small I-JEPA run that collapses outright under the default recipe. Our risk is the third kind, which no generic regulariser targets: a variance hinge on per-channel std is satisfied by position variation. Recommendation: no regulariser in the baseline, the diagnostics below from step 0, and a VICReg variance term or SIGReg as a prototype alternative.

## Small scale

- **I-JEPA paper:** no small-image results. ViT-B/16 at 224 px is its smallest model. It says I-JEPA converges about 5× faster in iterations than MAE.
- **C-JEPA Table 9:** ViT-T/16 and ViT-S/16 on IN-1K for 100 epochs reach only 22–32 % linear probe. Tiny ViTs with short JEPA training are weak at *semantic* classification.
- **Community runs on 32 px images:**
  - [keon/jepa](https://github.com/keon/jepa): a single-file I-JEPA. ViT 32 px, patch 4 (8 × 8 = 64 tokens, the same grid as ours), dim 128, depth 6, 4 heads. Predictor dim 64, depth 4. The I-JEPA multi-block sampler with block sizes per batch. EMA 0.996 → 1.0, AdamW lr 3e-4, wd 0.05, batch 256, smooth-L1. About 52.7 % CIFAR-10 linear probe after 100 epochs.
  - [HugoPlanchat/self-supervised-to-jepa](https://huggingface.co/HugoPlanchat/self-supervised-to-jepa): ViT-Tiny patch 4, d = 128, 6 layers, 1.2 M params, STL-10 for 100 epochs. CIFAR-10 linear: I-JEPA 56.3 %, LeJEPA 62.0 %, MAE 49.1 %, VICReg 73.1 %, BYOL 73.8 %.
  - Both runs train cleanly; neither reports collapse.
- **[TU Delft thesis "Reducing Data in Visual AI: I-JEPA"](https://repository.tudelft.nl/record/uuid:34c9347e-f34d-4def-a93b-1ce9cb0ee669):** ViT-Tiny/8 on 64 × 64 Tiny-ImageNet subsets, a 64-token grid identical to ours. The site refused automated fetches, so I could not read its numbers.
- **What carries over:**
  - The 8 × 8-token, width 128–192, depth-6 ViT is a well-trodden small JEPA configuration. It trains without tricks.
  - Our task is localisation of one rigid object, not semantics. Clevr/Count and Clevr/Dist (Table 4) are the closest I-JEPA evidence, and there I-JEPA beats DINO/iBOT on depth by a large margin. It keeps spatial, low-level information better than view-invariance methods.
- **Training length:**
  - Published small runs use 100 epochs of 50–100 k images, i.e. 5–10 M samples seen.
  - Our 1.45 M train Frames are near-duplicates of one Track, so samples seen is the right budget unit.
  - **Guess:** 20–40 k steps at batch 128 (2.5–5 M Frames). Ticket 03 measures Frames/s, so this is not yet in wall-clock time.
- **Known failure modes at small scale:**
  - Context too small (Table 9).
  - Too few targets (Table 10).
  - Target blocks too large or too small (Table 8).
  - High-frequency or noisy details ignored by design. Here that would be the car, if it is not predictable from the context.

## Output used downstream

- **What I-JEPA evaluates on:** the target encoder's last layer, average-pooled. For VISSL transfer, it picks the better of last-layer pooled and the concatenation of the last 4 layers pooled (App. A.2). There is no `[cls]` token.
- **Pooling is wrong for us:** it destroys the car's position. Our probe must read the **un-pooled 8 × 8 grid** (flattened 8·8·C), and the same holds for the ticket-01 probe on `vae-v1` Latents.
- **Precedent for generative models on frozen JEA patch features:**
  - [RAE, 2510.11690](https://arxiv.org/abs/2510.11690): diffusion transformers trained directly in frozen DINO/SigLIP/MAE patch-feature space.
  - V-JEPA 2-AC: predicts frozen V-JEPA 2 patch features.
  - Both use the last-layer patch tokens.
- **Feature dimension:** I-JEPA has no bottleneck; targets are the full width. Our PLAN wants about 32 channels, which leaves two options:
  - (a) **A linear 192 → 32 head at the end of the encoder, and the JEPA loss on the LayerNorm'd 32-d output.** The kept features are exactly the trained target space.
  - (b) Train with 192-d targets and reduce to 32 by PCA on frozen features. This keeps the I-JEPA recipe exact; PCA picks the high-variance directions, which may not be the car.
  - I recommend (a) as the baseline and (b) as an alternative (**guess**; no source compares them).
- **Normalisation:**
  - Targets are LayerNorm'd per token: zero mean and unit variance across channels at each patch.
  - The WAM still needs **per-channel** mean and std over train Frames, as for `vae-v1`, because the token-wise LayerNorm does not standardise each channel across Frames.
  - Store features after the token LayerNorm. Compute per-channel statistics over all 64 positions × a fixed Frame set, and record them in the Checkpoint card.

## Recommended starting recipe for ticket 03

Baseline `jepa-v1`:

**Context encoder**
- ViT on 8 × 8 px patches (64 tokens), width 192, depth 6, 3 heads (head dim 64), MLP ratio 4, pre-norm, final LayerNorm, 2D RoPE.
- No `[cls]` token; no absolute position (RoPE only).
- Head: Linear 192 → 32.
- About 2.7 M parameters.

**Target encoder**
- EMA copy of the context encoder; it sees the full Frame.
- Targets = `layer_norm(head(out), (32,))` without affine, at the target tokens.
- EMA momentum 0.996 → 1.0, linear over all steps.

**Predictor**
- Width 96, depth 4, 3 heads (head dim 32), 2D RoPE.
- Input projection 32 → 96; output projection 96 → 32.
- Shared learnable mask token (zero-init), with RoPE at the target positions.
- One pass per target block.
- About 0.45 M parameters.

**Masking (8 × 8 grid)**
- Block shape drawn per batch, positions per Frame.
- 4 target blocks, each 3 × 3, 3 × 4 or 4 × 3 (scale ≈ 0.15–0.2, aspect 0.75–1.5). Target blocks may overlap.
- Context: one block of scale 0.85–1.0, minus the targets; min 8 tokens.
- Batch masks are trimmed to the minimum counts, as in the official collator.
- **Car bias, partial:**
  - Per Frame, with probability 0.5, place one target block so that it covers ≥ 1 but not all of the car's patches.
  - The other target blocks are placed randomly, but rejection-sampled so that at least 1 car patch stays in the context.
  - For Frames whose car touches a single patch, fall back to random blocks.
  - Car patches come from the State (rotated rectangle → touched patches).
  - The remaining 50 % of Frames are plain random multi-block, which keeps background and colour learning.

**Loss**
- Mean squared error over target tokens and channels (PLAN). Smooth-L1 (official code) is an equivalent alternative.
- No anti-collapse term in the baseline.

**Optimisation (guesses scaled from the small-scale runs and the official recipe)**
- AdamW, betas (0.9, 0.95), batch 128.
- LR: peak 5e-4, linear warmup over the first 5 % of steps from 1e-4, cosine to 1e-6.
- Weight decay 0.04 → 0.4 (cosine, official), none on biases and norms.
- Grad clip 1.0 (V-JEPA uses 10).
- 30 k steps (3.8 M Frames); log diagnostics every 1 k steps.
- No data augmentation (I-JEPA uses none). Frames come from `frame_dataset`.

**Output kept:** the target encoder's `layer_norm(head(·))` grid, 8 × 8 × 32, then per-channel normalisation for the WAM.

Alternatives worth one prototype run each (same budget; score with the probe and the diagnostics):
1. **Car-bias strength:** car fraction 0 (pure I-JEPA), 0.5 (baseline), and "full-cover" (PLAN as written: ≥ half the blocks contain all car patches). This directly tests the surprise above; I expect full-cover to give a worse probe than partial.
2. **Anti-collapse safety net:** baseline plus a VICReg variance hinge on the per-channel std across the batch of the predictor's car-patch outputs, coefficient ~0.01–0.04 (**guess**). Or baseline plus SIGReg (λ = 0.05, 256 slices) on the context-encoder tokens. Try this if the diagnostics show the car signal decaying.
3. **Feature width:** loss on 192-d targets, then PCA → 32 (option b). Or 4 smaller target blocks (scale 0.10–0.15, 3 × 3 / 3 × 2), which leaves about 27 context tokens instead of 21. Useful if the prediction task looks too hard (high final loss, slow probe gains).

## Collapse diagnostics for ticket 04

Compute on a fixed evaluation set (e.g. 4096 Frames from validation episodes, fixed seed) using the **target** encoder's output f ∈ R^{N×8×8×32}, every 1 k steps and at the end. All thresholds are **guesses** to be calibrated on the ticket-03 prototype (log them first, gate later).

1. **Global per-channel std:** the std of each channel over all N·64 tokens. Catches complete collapse. After the token LayerNorm, healthy values are around 0.5–1. *Flag if* min over channels < 0.1. Note that a position-only encoder passes this.
2. **Across-Frame std at fixed position:** for each (position, channel), the std over N; then the mean over positions, min and median over channels. It measures how much the features vary with the Frame rather than the position. Expect it to be much smaller than item 1 on a fixed Track. Track its trend; *flag if* it falls monotonically for several kiloSteps.
3. **Car contrast:** for each Frame, take the patches the car touches. Compute d = ‖f(p) − μ_bg(p)‖ / √(tr Σ_bg(p)), where μ_bg and Σ_bg are the mean and covariance at position p over eval Frames whose car is ≥ 2 patches away. Average over car patches. This is the most direct "features encode the car" signal. *Flag if* < 2, or if it decays over training.
4. **Effective rank (RankMe):** exp(entropy of normalised singular values) of the centred token matrix [N·64, 32]. Max is 32. *Flag if* < 8, or if it drops sharply. Also compute it on the position-centred matrix (subtract μ(p)), which measures Frame-dependent dimensions only.
5. **In-training linear probe:** closed-form ridge from the flattened 2048-d grid to (x, y, sin h, cos h) on 2 k train / 2 k eval Frames, every 1–2 k steps. Report position p95 (m) and heading p95 (°). This is the real signal. The gate itself uses ticket 01's probe.
6. **Loss split:** JEPA loss on target tokens that touch the car vs those that don't, logged separately. Background loss → ≈ 0 is expected. If the car-token loss also → 0 while the car contrast falls, the car has been dropped from the targets.
7. **Predictor output std:** the V-JEPA `reg_fn` quantity, √var over patches of the predictor outputs. Near 0 means the predictor emits a constant.

## Sources

- I-JEPA paper: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023, https://arxiv.org/abs/2301.08243. Sections 3 and 9, App. A.1–A.2, Tables 6–14.
- I-JEPA code and configs: https://github.com/facebookresearch/ijepa: `configs/in1k_vith14_ep300.yaml`, `in22k_vitg16_ep44.yaml`, `src/masks/multiblock.py`, `src/train.py`, `src/helper.py`.
- V-JEPA code: https://github.com/facebookresearch/jepa: `configs/pretrain/vitl16.yaml`, `app/vjepa/train.py` (`reg_fn`, `loss_exp`).
- V-JEPA 2: Assran et al. 2025, https://arxiv.org/abs/2506.09985 (RoPE, L1 loss, fixed EMA).
- C-JEPA: Mo & Tong, NeurIPS 2024, https://arxiv.org/abs/2410.19560.
- LeJEPA: Balestriero & LeCun 2025, https://arxiv.org/abs/2511.08544; code https://github.com/rbalestr-lab/lejepa.
- Littwin et al., "How JEPA Avoids Noisy Features", NeurIPS 2024, https://arxiv.org/abs/2407.03475.
- RankMe: Garrido et al., ICML 2023, https://arxiv.org/abs/2210.02885.
- LiDAR: Thilak et al., ICLR 2024, https://arxiv.org/abs/2312.04000.
- RAE: Zheng et al. 2025, https://arxiv.org/abs/2510.11690.
- Masking precedents: AttMask https://arxiv.org/abs/2203.12719; SemMAE https://arxiv.org/abs/2206.10207; ADIOS https://arxiv.org/abs/2201.13100; masking for contrastive ConvNets https://arxiv.org/abs/2309.12757.
- Small-scale runs: https://github.com/keon/jepa (`ijepa.py`); https://huggingface.co/HugoPlanchat/self-supervised-to-jepa; TU Delft thesis https://repository.tudelft.nl/record/uuid:34c9347e-f34d-4def-a93b-1ce9cb0ee669 (not readable from here).
