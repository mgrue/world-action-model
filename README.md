# Toy world action model

A from-scratch, laptop-sized version of a FLUX-style **world action model**: a model that looks at a picture of the world, is told what to do, and produces the next few actions while also imagining what the world will look like.

Everything is a toy, small enough to train on a 16 GB Apple Silicon Mac:

- **`toycar`**: a deterministic 2D car simulator with a 64 × 64 overhead camera, a track that forks into two branches, a scripted expert driver, and a dataset recorder.
- **`wam`**: a small convolutional VAE that compresses each picture to an 8 × 8 × 16 Latent, and an 8.2 M-parameter transformer that learns by **flow matching** to generate 16 actions and 8 future pictures from one picture, the car's State and an Instruction (`outer`, `inner` or `any`).

The point of the exercise is the Fork. With Instruction `any` the expert takes either branch. A plain regression model averages the two and steers into the island between them; a flow-matching model should commit to one branch per sample.

The full plan is in [`PLAN.md`](PLAN.md); the vocabulary (Frame, State, Window, Latent, Chunk, …) is defined in [`CONTEXT.md`](CONTEXT.md).

## Status

| PLAN Step | What | State |
| --- | --- | --- |
| 1 | Simulator, keyboard driving | done |
| 2 | Scripted expert with Instructions | done |
| 3 | Dataset of expert episodes and training Windows | done |
| 4 | Frame compressor `vae-v1` | done, Checkpoint committed |
| 5–8 | World action model: Instruction encoding, transformer, flow-matching training, sampler | done, Checkpoint `wam-v1` committed |
| 9 | Closed-loop driving in the simulator | implemented; `wam-v1` gate run: FAIL (30.8 % crashes, 3.36 laps; Fork gates pass), see [`docs/step9-closed-loop.md`](docs/step9-closed-loop.md) |
| 10–11 | Experiments, comparison with the expert | not started |

## Documentation

Plain-language explainers, for readers without a machine-learning background:

- [`docs/step4-vae.md`](docs/step4-vae.md): the VAE frame compressor, what it does, how it is trained and checked.
- [`docs/step5-8-wam.md`](docs/step5-8-wam.md): the main model, flow matching, how it is trained and the checks it must pass.
- [`docs/step9-closed-loop.md`](docs/step9-closed-loop.md): letting the model drive, the replanning Scheme, the gates and what `wam-v1` does.

The exact, buildable specifications behind each Step live in `.scratch/` (one folder per Step, each with a `SPEC.md`, the decision tickets and the prototypes that measured things):

- [`.scratch/step1-simulator/SPEC.md`](.scratch/step1-simulator/SPEC.md)
- [`.scratch/step2-expert/SPEC.md`](.scratch/step2-expert/SPEC.md)
- [`.scratch/step3-dataset/SPEC.md`](.scratch/step3-dataset/SPEC.md)
- [`.scratch/step4-vae/SPEC.md`](.scratch/step4-vae/SPEC.md)
- [`.scratch/step5-8-model/SPEC.md`](.scratch/step5-8-model/SPEC.md)
- [`.scratch/step9-closed-loop/SPEC.md`](.scratch/step9-closed-loop/SPEC.md)

## Requirements

- macOS on Apple Silicon (training uses PyTorch's MPS backend; everything also runs on the CPU, just much slower).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.14 and every dependency.

## Setup

```sh
uv sync                      # Python 3.14, numpy, pygame-ce, torch, einops, pytest
uv sync --group viz          # optional: matplotlib, for plots and contact sheets
uv run pytest                # all tests, CPU; about 2 minutes
```

Tests that need the recorded dataset or a trained Checkpoint skip themselves until those exist.

Commands that draw pictures (`plot`, `show`, contact sheets) need the `viz` group: either `uv sync --group viz` once, or prefix the command with `uv run --group viz`.

## Build and run, Step by Step

Each Step leaves something you can check before moving on. Long runs are wrapped in `caffeinate -i` so the Mac doesn't sleep.

### Step 1: drive the car yourself

```sh
uv run python -m toycar.drive [--seed N] [--direction cw|ccw] [--route outer|inner] [--speed V] [--record DIR]
```

Keys: ↑/W throttle, ↓/S brake, ←/A →/D steer, H hi-res, Tab info panel, P pause, R next seed, Shift+R same seed, Esc/Q quit.

Replay a recording made with `--record`:

```sh
uv run python -m toycar.replay FILE.npz [--show] [--export-png DIR]
```

### Step 2: watch and score the scripted expert

```sh
uv run python -m toycar.expert show [--seed N] [--instruction outer|inner|any] [--direction cw|ccw] [--clean]
uv run python -m toycar.expert eval [--n 1000] [--instruction ...] [--direction ...] [--clean] [--workers K]
```

`show` opens a window with the expert driving; `eval` runs many episodes and reports laps, crashes and whether the Instruction was followed. `--clean` turns off the expert's deliberate steering perturbations.

### Step 3: record the dataset (required for everything below)

The dataset is not in the repository (`data/` is gitignored). Record it once:

```sh
uv run python -m toycar.dataset record          # ≈ 1 min, ≈ 100 MB in data/step3-v1/
uv run python -m toycar.dataset verify          # re-renders and re-simulates to check it
uv run python -m toycar.dataset stats           # episode and Window counts
uv run --group viz python -m toycar.dataset plot --split val --seed 0 --out window.png   # look at one Window
```

Recording is deterministic: the same code always produces the same bytes. Pictures are not stored; they are re-rendered from the stored States when needed.

### Step 4: the frame compressor `vae-v1`

`checkpoints/vae-v1/` is committed, so **you can skip training** and go straight to Step 5–8. To check it, or to retrain from scratch:

```sh
uv run python -m wam.compressor eval vae-v1                                   # car-visibility check; exit 0 on PASS
uv run --group viz python -m wam.compressor show vae-v1 --out contact.png     # originals vs reconstructions

# retraining (only if you want to reproduce it; ≈ 1 h 35 min on MPS)
caffeinate -i uv run python -m wam.compressor train --name vae-v1
uv run python -m wam.compressor promote runs/vae/<run> --name vae-v2         # Checkpoint names are never reused
```

Loading `vae-v1` prints a warning that `src/wam/compressor.py` changed since it was trained. That is expected: shared helpers were moved out of that file afterwards. The weights are unaffected.

### Steps 5–8: the world action model `wam-v1`

```sh
# 1. quick Run: 3,200 steps, ≈ 20 min. Its "Step 7 loss check" line must say PASS.
caffeinate -i uv run python -m wam.policy train --quick --name quick

# 2. full Run: 32,000 steps, ≈ 3 h. Exit 0 means every check passed.
caffeinate -i uv run python -m wam.policy train --name v1

#    if it is interrupted, continue exactly where it stopped:
caffeinate -i uv run python -m wam.policy train --resume runs/wam/<run>

# 3. look at runs/wam/<run>/samples.png, then freeze it and commit checkpoints/wam-v1/
uv run python -m wam.policy promote runs/wam/<run> --name wam-v1

# 4. re-check the frozen Checkpoint
uv run python -m wam.policy eval wam-v1 [--json out.json] [--sweeps]
uv run --group viz python -m wam.policy show wam-v1 --out samples.png
```

`eval` and `show` take either a Checkpoint name or a Run directory. Every Run writes `config.json`, `log.csv` (picture and action losses logged separately), `metrics.json`, `stdout.txt`, `samples.png` and `checkpoint.pt` to `runs/wam/<date-time>-<name>/`. `promote` refuses a Run that failed its checks unless you pass `--force`.

The training switches for the Step 10 experiments go through `--set`:

```sh
uv run python -m wam.policy train --name reg   --set objective=regression --set video_stream=false
uv run python -m wam.policy train --name novid --set video_stream=false
uv run python -m wam.policy train --name h8    --set horizon=8
uv run python -m wam.policy train --name nocfg --set p_uncond=0
```

Other switches: `video_loss_weight`, `action_loss_weight`, `t_shift`, `future_stride` (see [spec §5.5](.scratch/step5-8-model/SPEC.md)). Invalid combinations exit with code 2.

### Using a trained Policy from Python

```python
import wam.policy

policy = wam.policy.load("wam-v1")                 # EMA weights + the frozen vae-v1, on MPS if available
result = policy.act(frame, state.features(), "outer", n_samples=1, seed=0, return_latents=True)
result.actions[0, 0]                               # (16, 2) steer, throttle in [-1, 1]
policy.compressor.decode(result.latents[0, 0])     # (8, 64, 64, 3) imagined future Frames
```

One call takes about 70 ms on an M1 Pro. Optional arguments: `n_steps` (default 10), `cfg_action` / `cfg_video` (classifier-free guidance, default off), `noise=(x_z, x_a)` (supply the initial noise yourself instead of `seed`).

Loading `wam-v1` warns that `src/wam/flow.py` changed since it was trained: Step 9 added the optional `noise=` entry to the sampler. The weights and the sampling maths are unchanged.

### Step 9: let the model drive

```sh
uv run python -m wam.closedloop eval expert                    # reference driver: all four gates PASS (≈ 15 s)
uv run python -m wam.closedloop eval saboteur:<name>           # zero-steer, averaged-fork, slow, wrong-branch, always-outer:
                                                               #   each fails exactly its gate(s)
caffeinate -i uv run python -m wam.closedloop eval wam-v1      # 120-episode gate run (≈ 46 min); exit 0 PASS, 1 FAIL
caffeinate -i uv run python -m wam.closedloop eval wam-v1 --sweep replan       # also: cfg, samples, steps (60 dev episodes per point)
caffeinate -i uv run python -m wam.closedloop eval wam-v1 --robustness         # 9 Stress conditions, 60 episodes each (≈ 3.5 h)
uv run python -m wam.closedloop show wam-v1 --imagine          # watch it live (pygame); Space pause, N next, Esc quit
uv run --group viz python -m wam.closedloop video runs/drive/<run> --episode K [--hires]   # PNG Frames + trajectory.png
```

`<driver>` is a Checkpoint name, a Run directory, `expert` or `saboteur:<name>`. Every `eval` also runs the clean expert on the same episodes as the reference row, and writes `config.json`, `episodes.jsonl`, `ticks.npz`, `metrics.json`, `report.md` and `stdout.txt` to `runs/drive/<date-time>-<name>/`. The four gates are: crashes ≤ 5 %, Fractional laps ≥ 3.5, Instruction followed ≥ 95 %, `any` outer fraction between 0.30 and 0.70. Only a default-Scheme, 120-episode gate run of `wam-v1` gets a PASS/FAIL verdict. Every other run prints the table with a reason instead. Scheme overrides: `--execute-k --n-samples --n-steps --cfg-action --cfg-video --select first|coherence --rho --seed-mode per_replan|per_episode`.

## Repository layout

```
src/toycar/          simulator (torch-free): track, physics, render, env, expert, dataset, drive, replay
src/wam/             models (PyTorch)
  compressor.py      Step 4 VAE: train / eval / show / promote, load / encode / decode
  carcheck.py        car-visibility check: is the car's position and heading readable in a Frame?
  runs.py            Run directories and provenance helpers
  model.py           ModelConfig, token layout, RoPE, transformer blocks, WAM network
  flow.py            action normalisation, noise levels, losses, Euler sampler, guidance
  train.py           training loop, EMA, validation, logging, resume
  evaluate.py        eval sets, Fork references, metrics, gates, contact sheet
  policy.py          Policy API (load / act / sample / promote) and the wam.policy CLI
  drive.py           Step 9 Drivers (Policy + Scheme, expert, Saboteurs), Stress conditions, the lockstep loop
  closedloop.py      Step 9 Evaluation sets, metrics, gates, sweeps, robustness report and the wam.closedloop CLI
tests/               pytest suite (CPU, tiny models and a tiny recorded dataset)
checkpoints/         promoted, frozen Checkpoints with their card.json (committed)
docs/                plain-language explainers
.scratch/            per-Step specs, decision tickets, prototypes and measurements
data/, runs/         recorded dataset, training Runs and closed-loop Runs (gitignored)
```

## Credits

The model design mirrors FLUX 3 Action (Black Forest Labs), shrunk to a toy. No code from it is imported.
