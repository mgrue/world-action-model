# Step 4: the VAE frame compressor

2026-11-04 · plain-language explainer. The exact specification is `.scratch/step4-vae/SPEC.md`. Words in *italics* on first use are explained in the [Glossary](#glossary) at the end.

## In one paragraph

The project's main model will learn to drive by watching pictures of the track. A picture (a *Frame*) is 64 × 64 pixels with 3 colours each: 12,288 numbers. That's a lot to predict, and almost all of it is the same track every time. Step 4 trains a *VAE*, a small neural network that squeezes each Frame into a much shorter summary of 1,024 numbers (a *Latent*) and can turn that summary back into a picture. The main model then works only with these summaries. The VAE is trained once, checked to confirm it keeps the car's position and direction readable, and then frozen so it never changes again.

## Why the VAE exists

Think of it as a very specialised image compressor, like a JPEG made just for this one track.

- **Less to predict.** The main model has to imagine the next 16 Frames ahead. Predicting 16 × 1,024 numbers is far easier and faster than 16 × 12,288 pixels.
- **Ready-made pieces for the main model.** The summary is an 8 × 8 grid. Each of the 64 cells describes a 5 m × 5 m patch of the track, and the main model (a *transformer*) reads those 64 cells as its input pieces (*tokens*).
- **One fixed tool for everything that follows.** Steps 6–10 all load the same frozen compressor by name, `vae-v1`, so they all see the world the same way. If a better compressor is ever built, it gets a new name (`vae-v2`) and everything after it is rerun.
- **Only the car really matters.** The track never changes; the car's position and heading are the information the model needs. So the compressor is judged on one question: after squeezing and unsqueezing, can you still tell where the car is and which way it points?

## How it is built

The VAE has two halves:

- The **encoder** shrinks the picture step by step: 64 × 64 → 32 × 32 → 16 × 16 → 8 × 8. Each step halves the width and height, while describing every position with more *channels* (more numbers per position).
- The **decoder** does the reverse and grows the 8 × 8 summary back into a 64 × 64 picture.

```
        ENCODER  (picture → summary)                 DECODER  (summary → picture)

  Frame 64×64, 3 colours                        rebuilt Frame 64×64, 3 colours
            │                                                  ▲
  32×32 grid,  64 numbers per cell              64×64 grid,  32 numbers per cell
            │                                                  ▲
  16×16 grid, 128 numbers per cell              32×32 grid,  64 numbers per cell
            │                                                  ▲
   8×8 grid,  128 numbers per cell              16×16 grid, 128 numbers per cell
            │                                                  ▲
   8×8 grid:  16 "best guess" numbers           8×8 grid, 128 numbers per cell
            + 16 "uncertainty" numbers                         ▲
            │                                                  │
            └──────►  LATENT: 8×8 grid × 16 numbers = 1,024  ───┘
```

Details, for reference:

- Each step uses *convolutions*, small filters that slide over the image. Each one looks at a 3 × 3 neighbourhood of cells.
- Between steps there are two *residual blocks*: a few extra filters whose result is added on top of their input, which makes deep networks easier to train.
- The whole network has 3,166,435 *parameters*. It is small by modern standards and trains on a laptop GPU (*MPS*) in about 1.5 hours.
- The 8 × 8 grid and the 16 numbers per cell are fixed, because later Steps depend on them. Only the internal sizes could change in a future `vae-v2`.

## How it learns

Training shows the VAE about 1.5 million track pictures, 128 at a time. For each one it compresses, rebuilds and compares the rebuilt picture with the original. A score called the *loss* says how wrong it was, and the network nudges its parameters to lower that score. This repeats 12,000 times.

The loss has two parts:

```
loss = (weighted L1 difference between rebuilt and original picture)  +  0.001 × KL
```

1. **Picture difference (*L1*), with extra weight on the car.** The car covers only about 0.7% of the picture. With plain L1 the network can almost ignore the car and still score well, and it does: it simply paints an empty track. So every car pixel counts 31 times as much as a background pixel (weight 1 + *λ*, with λ = 30). Then the car matters about as much as the whole background together.
2. **A small *KL* term.** It keeps the summaries tidy and well-behaved, so slightly wrong summaries still decode into sensible pictures. This matters later, because the main model's predicted summaries will never be perfect. Its weight is tiny (0.001) so it doesn't fight the picture quality.

After training, the 16 numbers in each cell are rescaled (*normalised*) so that each is centred on 0 and spread with a standard deviation of 1 over the training pictures. That's a convenient, consistent scale for the main model.

## How we know it works: the car-visibility check

After training, the VAE compresses and rebuilds 4,096 pictures it has never seen (the *validation* set). A separate program estimates the car's position and heading in each original and each rebuilt picture, and compares them.

To **pass**:

- the car must be found in every single rebuilt picture;
- position may be off by at most 0.3 m in 95% of pictures (*p95*) and at most 0.6 m in the worst one. One pixel is 0.625 m;
- heading may be off by at most 10° in 95% of pictures and at most 30° in the worst one.

It must also still pass when a little random noise is added to the summaries. That simulates the main model's imperfect predictions.

## Why this particular design

Five trial versions, called runs A–E, were trained before the design was fixed. Each changed one thing:

| Run | What changed | Size | Training steps | Position error, 95% / worst (m) | Heading error, 95% / worst (°) | Result |
| --- | --- | --- | --- | --- | --- | --- |
| A | plain L1, no KL | 1.84 M | 6,000 | car found in only 14.5% | 180 / 180 | FAIL: car erased |
| B | + extra weight on the car | 1.84 M | 6,000 | 0.21 / 0.39 | 12.2 / 28.6 | FAIL |
| C | + small KL (0.0001) | 1.84 M | 6,000 | 0.19 / 0.43 | 11.5 / 38.0 | FAIL |
| D | KL 0.001 | 1.84 M | 6,000 | 0.18 / 0.36 | 10.9 / 31.7 | FAIL |
| **E** | bigger network, trained twice as long | **3.17 M** | **12,000** | **0.12 / 0.26** | **8.0 / 24.7** | **PASS** |

What they taught us:

- **A:** without extra car weight, the network simply erases the car.
- **B–D:** position was fine, but heading missed the limit by about a degree. The yellow nose that shows the direction came out slightly blurred.
- **E:** a bigger network, trained longer, draws the nose sharply enough. It passes, also with noise. Its recipe became the specification, and the current `vae-v1` training run repeats it.

Known weak spots, recorded but accepted:

- **Cars off the road** are rebuilt less accurately, because there are few such pictures to learn from.
- **Unusual colours** (e.g. brown grass, a blue car) are repainted in the familiar training colours. These are the *Probes* used later in Step 10, so some Probe failures may start here.

## How later Steps use it

Once `vae-v1` passes, it is saved ("promoted") to `checkpoints/vae-v1/` and committed. Later Steps only ever call:

| Call | What it does | Used by |
| --- | --- | --- |
| `load("vae-v1")` | loads the frozen compressor | Steps 6–10 |
| `encode(frames)` | pictures → summaries (8 × 8 × 16 each) | Steps 6–7 training, Step 9 live driving, Step 10 Probes |
| `encode_windows(batch)` | the same, for one *Window* of training data (the current Frame plus the next 16) in one go | Steps 6–7 training |
| `decode(z)` | summaries → pictures, e.g. to look at what the main model imagines | Step 8 |

Pictures are compressed on the fly when needed (about 6 ms per Window), so no summaries are stored on disk. The same car-visibility check is reused in Step 8 to judge the main model's imagined pictures.

## Glossary

**Channel**: one of several numbers stored per position in a grid. A colour picture has 3 channels (red, green, blue); the Latent has 16 per cell.

**Checkpoint**: a saved, frozen copy of trained network weights with a record of how they were made (`card.json`). Referred to by name, e.g. `vae-v1`, and never overwritten.

**Convolution (conv)**: a small filter, here 3 × 3, that slides over a grid and computes new numbers from each neighbourhood. It's the basic building block of image networks. A "stride-2" convolution moves 2 cells at a time, which halves the grid size.

**Decoder / encoder**: the two halves of the VAE. The encoder turns a picture into a Latent; the decoder turns a Latent back into a picture.

**Flow matching**: the method Step 7 will use to train the main model to imagine future Latents. Mentioned here only because it prefers normalised Latents.

**Frame**: one 64 × 64 colour picture of the whole track, seen from above, one per simulation tick (15 per second).

**KL (Kullback–Leibler divergence)**: a measure of how different two probability distributions are. In a VAE, the encoder produces for every Latent number a *best guess* and an *uncertainty*. The KL term measures how far these stray from a plain standard bell curve (centre 0, spread 1), and penalises straying. Effect: the summaries stay compact and smooth, with no wild values and no gaps, so nearby summaries decode to similar pictures. A small weight (0.001) gives this smoothness without hurting picture quality.

**L1 loss (mean absolute error)**: the average of the absolute differences between two sets of numbers. For pictures: take each pixel of the rebuilt picture, subtract the original, drop the minus sign, and average. Example: original values (100, 50), rebuilt (90, 55) → differences 10 and 5 → L1 = 7.5. Compared with squared differences (L2), L1 is less forgiving of many small blurry errors, so it gives sharper pictures.

**Latent**: the compressed summary of one Frame: an 8 × 8 grid with 16 numbers per cell, 1,024 numbers in total. The main model reads and predicts Latents instead of pixels.

**Loss**: a single number saying how wrong the network currently is. Training means repeatedly adjusting the parameters to make it smaller.

**λ (lambda)**: the extra weight for car pixels in the loss, here 30. Car pixels count 1 + λ = 31 times as much as background pixels.

**MPS**: Apple's GPU interface on Mac. Training runs on it about 11× faster than on the CPU.

**Normalisation**: rescaling numbers so they share a common scale. Here each of the 16 Latent channels is shifted and stretched to have mean 0 and standard deviation 1 over the training pictures.

**p95 (95th percentile)**: the value that 95% of measurements stay at or below. "Position p95 ≤ 0.3 m" means that in at least 95% of pictures, the car's position is off by 0.3 m or less.

**Parameters (weights)**: the adjustable numbers inside a network that training tunes. This VAE has about 3.2 million.

**Probe**: a deliberate change to the world (other colours, a different track shape) that never appears in training. Used in Step 10 to test how well the model generalises.

**Residual block**: a small group of layers whose output is added to its own input ("skip connection"). Each block only has to learn a correction, which makes deeper networks train reliably.

**Run**: one training attempt, with its logs and results, stored under `runs/vae/`. Only a Run that passes the check is promoted to a Checkpoint.

**Token**: one input piece for a transformer. Here, one cell of the 8 × 8 Latent grid: 64 tokens per Frame, 16 numbers each.

**Transformer**: the type of neural network used for the main model (Step 6). It reads a list of tokens and learns how they relate to each other.

**VAE (variational autoencoder)**: an *autoencoder* is a network trained to compress its input and rebuild it. "Variational" means the encoder outputs a best guess plus an uncertainty for each number, and training adds a little random noise within that uncertainty. Together with the KL term, this keeps the compressed space smooth. After training, only the best guess is used.

**Validation (val) set**: pictures held back from training and used only to check the model, so the check measures how it does on pictures it has not memorised.

**Window**: one training example for the main model: the current Frame, the car's state, the instruction, and the next 16 actions and Frames.
