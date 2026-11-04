# Toy world action model: build plan

Goal: build a very small world action model from scratch, in simulation only, to understand the
full pipeline end to end. The design mirrors FLUX 3 Action (Black Forest Labs), with every
component shrunk to a toy that trains on a 16 GB Apple Silicon Mac.

Each step leaves something working that can be checked before moving on.

## Part 1: the world and the data

### 1. Simulated task
- 2D top-down car on a track, seen from a fixed overhead camera.
- Render 64 x 64 RGB frames with numpy or pygame.
- State: position, heading, speed. Action: steer, throttle.
- Deterministic physics: a seed reproduces an episode exactly.
- Make the car visually distinctive, with a marker on its nose so heading is readable.

**Check:** a human can drive the car with the keyboard and the rendering looks right.

### 2. Scripted expert
- Pure-pursuit controller following the track centerline.
- At least two distinct behaviors that are both sensible, for example:
  - "drive clockwise" vs "drive counterclockwise", or
  - "take the inner lane" vs "take the outer lane".
- These give instructions to condition on, and create situations where two different
  actions are both correct (the reason for using flow matching instead of regression).

**Check:** the expert completes laps reliably for every behavior.

### 3. Dataset
- Record a few thousand episodes: frames, states, actions, instruction label.
- Randomize start position, speed and minor visuals.
- Cut episodes into training windows:
  - 1 observed frame
  - the next 16 frames
  - the current state
  - the next 16 actions
  - the instruction

**Check:** plotting a window shows frames and actions that line up in time.

## Part 2: the model

### 4. Frame compressor (toy video VAE)
- Small convolutional autoencoder: 64 x 64 frame to an 8 x 8 grid of 16-channel latents.
- Train on recorded frames, then freeze it.

**Check:** reconstructions keep the car's position and heading clearly visible.

### 5. Instruction encoding
- With a handful of fixed instructions, use one learned embedding per instruction.
- Optional later: swap in a small pretrained text encoder for free-form wording.

### 6. Transformer with streams
- About 5 to 20 million parameters.
- One token sequence per window:
  - observed-frame latent tokens (clean)
  - one state token (clean)
  - one instruction token
  - future-frame latent tokens (noised)
  - action tokens, one per future step (noised)
- Separate input projection and output head per token type.
- Time position on every token, so action k and frame k line up.
- Noise level injected through adaptive layer norm (shift, scale, gate), as in the FLUX blocks.
- Start with one shared stack of blocks. Per-stream blocks can come later.

### 7. Flow-matching training

```
t      ~ distribution over [0, 1], one per window
x_t    = t * noise + (1 - t) * clean
loss   = mse(model(x_t, t, conditioning), noise - clean)
```

- Applied to future-frame latents and actions together.
- Weight the action loss up so a few action tokens are not drowned out by many video tokens.

**Check:** training loss falls for both streams; logged separately.

### 8. Sampler
- Start future frames and actions from pure noise.
- About 10 Euler steps along the predicted velocity.
- Read out the actions. Optionally decode the predicted frames to see what the model imagines.

**Check:** sampled action chunks look like the expert's on held-out windows.

## Part 3: close the loop and learn from it

### 9. Closed-loop evaluation in the simulator
- Each tick: feed current frame, state and instruction.
- Execute the first few predicted actions, then replan.
- Metrics: laps completed, crashes, instruction followed.

### 10. Experiments
- **Plain regression baseline:** predict actions directly with an MSE head. At a fork where both
  routes are valid it should average into the wall; the flow model should commit to one route.
- **No video loss:** drop the future-frame target. Does predicting the future help? With how much data?
- **Classifier-free guidance:** blank the instruction ~10% of the time in training; at sampling, push
  away from the unconditioned prediction and measure instruction-following.
- **Chunk length and replanning rate:** how far ahead can the model plan before execution drifts?
- **Small world changes:** change track color or shape and see what generalizes.
- **JEPA future targets:** swap the VAE for a self-supervised JEPA encoder and see whether
  predicting the *meaning* of future frames, not their pictures, helps the model drive.
  The transformer, flow matching, data, seeds, step count and gates all stay the same; only
  the frame encoder changes. Details in 10a.

#### 10a. JEPA encoder and comparison
- **Encoder `jepa-v1`** (I-JEPA style, built like Step 4: its own spec, Checkpoint and explainer):
  - context encoder: small ViT over 8 x 8-pixel patches (64 tokens), 2D RoPE, width ~192,
    ~6 blocks; outputs an 8 x 8 grid of ~32 features per frame (the part we keep);
  - target encoder: slow EMA copy of the context encoder, sees the whole frame, no gradients;
  - predictor: narrow transformer that predicts the target features of hidden patches from the
    visible ones; discarded after training;
  - masking: hide a few rectangles per frame, at least half of them covering the car (its
    position is known from the State), so the features must encode the car;
  - loss: squared error to the target features at hidden patches; no pixel reconstruction.
- **Check (no decoder, so a probe instead of pictures):**
  - a linear probe on frozen features reads car x, y and heading: position ≤ 0.3 m p95 and
    heading ≤ 10° p95, the bar `vae-v1` passed;
  - no collapse: every feature channel keeps a non-trivial spread across frames (logged in training);
  - report only: the same probe on the colour-randomised Probe frames.
- **WAM switch** `future_target=vae|jepa`: future frames become JEPA feature grids, normalised per
  channel and grouped 2 x 2 into 16 tokens each, so the 162-token layout stays; only the picture
  input and output layers widen.
- **Runs to compare** (same seeds and recipe as `wam-v1`):
  - `wam-v1`: VAE observed frame, VAE future targets (baseline);
  - JEPA future targets, VAE observed frame (isolates the target);
  - JEPA for both observed frame and future targets;
  - optionally each with `video_stream=false`, to see whether imagining the future helps and
    whether that depends on the encoder.
- **Comparison:**
  - action gates (rmse, commit, follow, balance, two-sided) and closed-loop metrics, compared directly;
  - imagined futures: train the same kind of linear probe on `vae-v1` Latents too and score both
    models' futures with their own probe; `carcheck` on decoded pictures stays an extra for the VAE.
- **Open questions** for the spec: patch size (8 or 4 pixels), feature size, car-masking ratio,
  EMA schedule and collapse diagnostics, probe thresholds.
- **Later, separate idea:** a pure JEPA world model (V-JEPA 2-AC style) that rolls features forward
  for candidate actions and plans by search instead of flow matching. A much bigger project; only
  worth it if the encoder swap shows JEPA features help here.

### 11. Compare against the expert
- The scripted controller is the upper bound.
- The gap shows the cost of imitation, and makes the comparison with classical control concrete.

## Budget and environment
- Steps 1 to 4: about a weekend.
- Steps 6 to 9: the core work, a few hundred lines of PyTorch.
- Runs on the Mac's GPU through PyTorch's Apple (MPS) backend.

## Reference code in flux-action
- `src/flux_action/processing/packing.py`: token packing, noising, loss.
- `src/flux_action/inference/sampling.py`: samplers and classifier-free guidance.
- `src/flux_action/models/transformer.py`: the multi-stream transformer.
- `examples/games/dataset.py`: a custom dataset plug for a game.
