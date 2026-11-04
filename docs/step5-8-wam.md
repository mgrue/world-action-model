# Steps 5–8: the main model and how it is trained

2026-11-04 · plain-language explainer. The exact specification is `.scratch/step5-8-model/SPEC.md`. Words in *italics* on first use are explained in the [Glossary](#glossary) at the end.

## In one paragraph

The main model is the "driver" of this project. It looks at one picture of the track (a *Frame*), the car's current position, heading and speed (the *State*) and a command (the *Instruction*: take the `outer` branch, the `inner` branch, or `any`). From these it produces the next 16 steering and throttle commands (a *Chunk*), and at the same time it *imagines* what the next 16 ticks will look like. It is a small *transformer* (8.2 million parameters) trained by *flow matching*. Flow matching means it learns to turn random noise, step by step, into a plausible Chunk and plausible future pictures. It learns from about a million examples of the scripted expert driving. Once it passes its checks, it is frozen as the Checkpoint `wam-v1`, and Step 9 lets it drive the car.

## Why predict actions *and* pictures?

This mirrors FLUX 3 Action, the much larger model this project copies in miniature. Two ideas matter:

- **Imagining the future helps choose the action.** To predict the next 16 pictures, the model has to understand how the car moves when it steers and accelerates. That understanding then also serves the actions. Whether it really helps is one of the Step 10 experiments: the same model is trained without the pictures and the two are compared.
- **Both are predicted together, in one pass.** Pictures and actions are generated side by side, and each can "see" the other while they take shape. So the imagined pictures and the planned actions stay consistent with each other.

The model doesn't predict real pixels. It predicts *Latents*: the compressed 8 × 8 × 16 summaries made by the frozen compressor `vae-v1` from Step 4. To look at an imagined picture, the Latent is turned back into pixels with `vae-v1`.

## Why flow matching instead of simply predicting the answer?

At the *Fork*, the track splits into two branches. With Instruction `any`, the expert sometimes goes left and sometimes right, and both are correct.

A plain prediction network (*regression*) learns to output the *average* of all correct answers. At the Fork, the average of "steer left" and "steer right" is "go straight", into the island between the branches. That is the classic failure this project is built to show.

Flow matching avoids it. It doesn't output one answer. It learns to *sample* an answer: start from random noise and gradually reshape it into something that looks like real expert driving. Different noise gives different, equally valid answers. At the Fork some samples go outer and some go inner, but none go down the middle.

```
  noise ──► a bit less noisy ──► … ──► clean Chunk (and clean imagined Frames)
  t = 1          t = 0.9                 t = 0
        10 small steps, each guided by the model's "which way to clean up" estimate
```

## What the model reads: one row of 162 tokens

A transformer reads a list of *tokens*, each a list of 256 numbers. Every training example (a *Window*) becomes exactly 162 tokens, always in this order:

| Tokens | Count | What they are | Clean or noisy |
| --- | --- | --- | --- |
| observed Frame | 16 | the current picture's Latent, cut into a 4 × 4 grid of patches (each 2 × 2 Latent cells, 10 m × 10 m of track) | clean |
| State | 1 | position, heading and speed | clean |
| Instruction | 1 | `outer`, `inner`, `any`, or the special *Null Instruction* | clean |
| future Frames | 8 × 16 = 128 | the imagined pictures at ticks 2, 4, …, 16 (every second one, to save work) | noisy |
| actions | 16 | steer and throttle for ticks 1 … 16 | noisy |

The first 18 tokens are the *conditions*: facts the model is given. The last 144 are what it has to produce. During generation they start as pure noise and are cleaned up step by step.

How the model knows which token is which:

- **Its own input layer per kind of token.** Picture patches, the State, the Instruction and actions each get their own small layer that turns them into 256 numbers. Observed and future picture patches share one layer, because they are the same kind of thing.
- **A position tag on every token** (*RoPE*): the tick it belongs to (0 for the present, 1–16 for the future) and, for picture patches, where on the 4 × 4 grid it sits. Action 6 and future picture 6 share the tick, so the model can connect "what I do at tick 6" with "what it looks like at tick 6".
- **The noise level.** Each token is told how noisy it currently is: 0 for the conditions, the current *noise level* t for the future pictures and actions.

Every token can look at every other token (*full attention*). There is no "only look at the past" rule. That's what lets the pictures and actions shape each other.

## The network itself

The design is FLUX's, shrunk down:

- **8 identical blocks**, each with 4 *attention heads*. In every block, each token gathers information from all the other tokens (*attention*) and then processes it on its own (a small feed-forward layer, *SwiGLU*). In FLUX's "parallel" layout both happen side by side and their results are added to the token.
- **Noise-level conditioning** (*adaLN*): before each block, every token is rescaled and shifted depending on its noise level and its kind. That's how the same blocks behave differently for a clean condition and a half-noisy action.
- **Two output layers**: one turns the future-picture tokens back into Latent patches, the other turns the action tokens into steer and throttle. Each output is the "which way to clean up" direction (the *velocity*), not the clean answer itself.
- **Safe start** (*adaLN-Zero*): at the start of training every block does nothing and the outputs are exactly zero. Training then gradually switches parts on, which keeps the start stable.

Size: 8,229,888 parameters in total, about 33 MB on disk. That's small by modern standards, but it took measuring to fit: on the Mac's GPU, attention gets expensive quickly as the number of tokens grows. That's why only every second future picture is predicted, and each picture is cut into 16 patches rather than 64.

## The Instruction, and "no Instruction"

Each of the three Instructions is one learned token: a row of 256 numbers that training adjusts. There is also a fourth row, the **Null Instruction**, which means "no Instruction given". It is not the same as `any`: `any` is a real command ("either branch is fine"), while Null means "I'm not telling you".

During training, the real Instruction is replaced by Null for 10% of the examples. So the model also learns to drive without being told where to go. That enables *classifier-free guidance* (CFG) later: run the model once with the Instruction and once with Null, then push the result further in the direction the Instruction makes a difference. That can make the model follow Instructions more strictly. CFG is off by default; Step 10 measures whether it helps.

## How it learns

### One training step

1. **Take 32 Windows** from the training set. Each has the current picture, the State, the Instruction, the next 16 expert actions (the labels) and the matching future pictures.
2. **Compress the pictures** with the frozen `vae-v1`.
3. **Pick a noise level** t between 0 and 1 for each Window. It is drawn more often near the noisy end, because the big decisions (which branch to take) are made there.
4. **Mix noise into the answers**: noisy = t × random noise + (1 − t) × true answer. At t = 1 it's pure noise, at t = 0 the true answer.
5. **Ask the model** for the clean-up direction, and compare it with the correct one, which is simply "noise minus true answer".
6. **Score it** (the *loss*) as the average squared error. Pictures and actions are averaged separately and then added, so the 32 action numbers per Window count as much as the 8,192 picture numbers.
7. **Adjust the parameters** a little to lower the loss.

Actions are rescaled first (*normalised*) so that steer and throttle each have spread 1, the same scale as the Latents. The model's raw outputs are converted back to real units, and clipped to [−1, 1], before anyone sees them.

### The full recipe

| | Full Run | Quick Run |
| --- | --- | --- |
| Steps × Windows per step | 32,000 × 32 (≈ 1 million Windows, each about 1.3 times) | 3,200 × 32 |
| Learning rate | 0.0003, warmed up over the first 1,000 steps, then slowly lowered to 5% | same |
| Time on the Mac (MPS) | ≈ 3 h | ≈ 20 min |

Other details:

- **The Windows are fully random** over the 797,262 training Windows. No situation is shown more often than it occurs in the data.
- **A smoothed copy of the weights** (*EMA*) is kept alongside training: a running average of the weights over roughly the last 1,000 steps. It is less jittery than the latest weights, and it's the copy that is evaluated and frozen.
- **Training can be resumed.** Every 1,000 steps a resume file is saved. After a crash or interruption, `train --resume` continues and ends up exactly where an uninterrupted run would have.
- **Everything is seeded.** The same seed on the same machine gives exactly the same result.
- **Pictures are drawn on the fly.** Every Window's pictures are rendered and compressed when needed, about 95 Windows per second in total. Nothing extra is stored on disk.

### What gets logged

Every 100 steps the Run's `log.csv` gets the training loss, **split into a picture part and an action part**. The action part is also split into "very noisy" and "nearly clean" examples. Every 2,000 steps the same losses are measured on 1,024 fixed *validation* Windows, which training never sees. Every 8,000 steps a small sampling test runs, so you can watch the driving quality improve.

## How we know it works: the checks

All checks use the smoothed (EMA) weights, 10 clean-up steps, no CFG and fixed seeds. The thresholds were fixed before any model was trained. They may be tightened once after the first quick Run, never loosened.

1. **The loss fell (PLAN Step 7).** For both pictures and actions, the final validation loss must be at most half its value at the start (about 2), and within 5% of the lowest value seen during the Run (so it didn't get worse at the end).
2. **The actions match the expert (PLAN Step 8).** On 951 validation Windows away from the Fork, one sampled Chunk per Window is compared with the expert's. The typical error (*RMSE*) must be at most 0.15 for steer and for throttle. For scale: predicting the average action scores 0.61 / 0.39, and the expert's labels with a bit of added noise score 0.10.
3. **At the Fork, the model commits to a branch.** On 512 validation Windows just before the Fork, 16 Chunks are sampled per Window. Each Chunk's steering is compared with two reference shapes: the expert's average "going outer" and "going inner" steering.
   - **Commit:** at least 90% of samples must be clearly closer to one of the two shapes. A Chunk that steers down the middle fails.
   - **Following:** with `outer` or `inner`, at least 90% of samples must take the commanded branch.
   - **Balance:** with `any`, between 30% and 70% of samples must go outer.
   - **Both ways:** with `any`, at least 80% of Windows must have samples going each way. A model that always picks the same branch fails.

   This test was first tried on fakes: the expert's own labels pass, while "average of the two shapes" and "random branch" fail.
4. **Reported, not gated:** how well the imagined pictures show the car in the right place (scored with the car-visibility check from Step 4), the best of 16 samples, the error per tick, and optionally sweeps over CFG strength and the number of clean-up steps.

A contact sheet `samples.png` shows it at a glance:

- true and imagined future pictures;
- sampled Chunks against the expert's;
- at the Fork, two bundles of `any` samples, one along each reference shape.

## How to run it

```
uv run pytest                                                            # all tests, CPU, includes tiny training runs
caffeinate -i uv run python -m wam.policy train --quick --name quick     # ≈ 20 min: Step 7 line must say PASS
caffeinate -i uv run python -m wam.policy train --name v1                # ≈ 3 h; exit 0 means every check passed
uv run python -m wam.policy promote runs/wam/<run> --name wam-v1         # freeze it, then commit checkpoints/wam-v1/
uv run python -m wam.policy eval wam-v1                                  # same verdict again
```

A Run lives in `runs/wam/<date-time>-<name>/`:

- `config.json`, `log.csv`, `metrics.json`, `stdout.txt`;
- `samples.png`;
- `checkpoint.pt` (the smoothed weights).

Only a Run that passes every check is promoted to `checkpoints/wam-v1/`, together with a `card.json` that records exactly how it was made.

## How later Steps use it

**Step 9 (driving the car in the simulator):**

```python
policy = wam.policy.load("wam-v1")
result = policy.act(frame, state.features(), "outer", seed=0)
actions = result.actions[0, 0]   # 16 × (steer, throttle); execute the first few, then ask again
```

One call takes about 70 ms on the Mac, including compressing the picture. How many of the 16 actions to execute before asking again is Step 9's choice.

**Step 10 (experiments):** everything it needs is a switch, so the same code trains the alternatives:

| Experiment | How |
| --- | --- |
| Regression instead of flow matching | `--set objective=regression --set video_stream=false` (expected to fail the Fork "commit" check: that's the point) |
| No imagined pictures | `--set video_stream=false` |
| Classifier-free guidance | `act(..., cfg_action=2.0)` or `eval --sweeps` |
| Shorter Chunks | `--set horizon=8` or `4` |

These variants are evaluated with the same checks but never have to pass them; they are saved as `wam-v1-<variant>` with `--force`.

## Glossary

**adaLN (adaptive layer normalisation)**: rescaling each token's numbers to a standard spread, then stretching and shifting them by amounts that depend on the token's noise level. That's how the network is told "this token is very noisy" or "this one is clean". **adaLN-Zero** starts those amounts at zero, so every block starts as "do nothing".

**Attention**: the transformer's core operation. Every token computes how relevant each other token is to it and takes a weighted mix of their information. An **attention head** is one independent copy of this; with 4 heads a token can look for 4 different kinds of relationship at once.

**Chunk**: the 16 actions (steer, throttle) the model produces for one Window. The executed actions come from its first few.

**Checkpoint**: a saved, frozen copy of trained weights with a record of how they were made (`card.json`), referred to by name (`vae-v1`, `wam-v1`) and never overwritten.

**Classifier-free guidance (CFG)**: run the model with and without the Instruction, take the difference, and amplify it: result = without + w × (with − without). w = 1 means no guidance; larger w follows the Instruction more strongly. It only works because training sometimes hid the Instruction (the Null Instruction).

**EMA (exponential moving average)**: a smoothed copy of the weights, updated every step as 99.9% old copy + 0.1% current weights. It averages out the step-to-step jitter of training and usually performs better.

**Euler steps**: the simplest way to follow the clean-up direction: take a small step in that direction, ask the model again, repeat. Here 10 steps go from pure noise (t = 1) to the answer (t = 0).

**Flow matching**: a way of training a model to generate samples. Real answers are mixed with random noise in varying amounts, and the model learns the direction that leads from the noisy mix back toward a real answer (the **velocity**). To generate, start from pure noise and follow that direction in small steps.

**Fork**: the point where the track splits into the outer and inner branch. The branches rejoin at the Merge.

**Frame**: one 64 × 64 colour picture of the whole track, seen from above, one per tick (15 per second).

**Instruction**: the command the driver follows: `outer`, `inner` or `any`. It decides which branch to take at the Fork.

**Latent**: the compressed summary of one Frame made by `vae-v1`: an 8 × 8 grid with 16 numbers per cell.

**Loss**: one number saying how wrong the model currently is. Training adjusts the parameters to make it smaller.

**Noise level (t)**: how much noise is mixed into an answer, from 0 (clean) to 1 (pure noise).

**Normalisation**: rescaling numbers to a common scale. Actions are shifted and stretched so steer and throttle each have spread 1.

**Null Instruction**: the "no Instruction given" token. Never in the data; swapped in for 10% of training examples and used for CFG. Not the same as `any`.

**Parameters (weights)**: the adjustable numbers inside the network that training tunes. This model has about 8.2 million.

**Regression**: predicting the answer directly in one go, trained to minimise the squared error. Where two different answers are both correct, it learns their average.

**RMSE (root mean square error)**: square every error, average them, take the square root. A typical size of the error, in the same units as the actions (−1 … 1).

**RoPE (rotary position embedding)**: a way of telling attention where each token sits (here: tick, grid row, grid column) by rotating its numbers by an angle that depends on the position. Attention then depends on how far apart two tokens are, not on where they are in absolute terms.

**Run**: one training attempt, with its logs and results, stored under `runs/wam/`. Only a Run that passes the checks is promoted to a Checkpoint.

**State**: the car's position, heading and speed. The model reads it as 5 numbers: x/40, y/40, cos(heading), sin(heading), speed/10.

**SwiGLU**: the per-token processing layer inside each block: two parallel linear layers, one of them passed through a smooth on/off curve (SiLU) that gates the other.

**Token**: one input piece for the transformer, a list of 256 numbers. Here: one 2 × 2 patch of a Latent, the State, the Instruction, or one action.

**Transformer**: the type of neural network used here. It reads a list of tokens and repeatedly lets them exchange information through attention.

**Validation (val) set**: Windows from episodes held back from training, used only for checking, so the checks measure how the model does on situations it hasn't memorised.

**Window**: one training example: the current Frame and State, the Instruction, the next 16 expert actions and the 16 Frames they cause.
