# Step 4 spec: `wam.compressor` frozen frame compressor (VAE) and the car-visibility check

Buildable spec for PLAN.md Step 4: a small convolutional VAE that maps a 64×64 Frame to an 8×8 grid of 16-channel **Latents**, trained on recorded train Frames, checked, then frozen as a named **Checkpoint**. It also fixes the **interface contract** that Steps 6–10 rely on: the encode/decode API and latent layout, latent normalisation for flow matching, the Checkpoint format and its provenance, and how Windows get their latents (encoded on the fly).

Sources: every decision here was made in a ticket of `MAP.md`, and each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root (Frame, State, Split, Window, Probe, Latent, Checkpoint, Run, …). The Step 3 contract is `.scratch/step3-dataset/SPEC.md`; this spec only adds to it. Measurements are in `research/01-torch-on-mac.md`; the prototypes that justified the design are `prototypes/vae_proto.py` and `prototypes/pose_estimator.py`, with their outputs in `prototypes/out/`. Section 14 lists clarifications made while compiling that weren't asked explicitly.

PLAN.md check: "reconstructions keep the car's position and heading clearly visible" — measured by the car-visibility check (§4) and seen on its contact sheet (§12).

---

## 1. Scope and layout

([ticket 02](issues/02-package-layout-and-checkpoints.md))

- New module **`wam`** ("world action model") in the **same uv project**, next to `toycar`:
  ```
  src/toycar/              unchanged, numpy only (one small additive API, §14.1)
  src/wam/__init__.py
  src/wam/compressor.py    VAE network, Compressor, load, promote, training loop, CLI (python -m wam.compressor)
  src/wam/carcheck.py      the car-visibility check (§4), independent of the compressor
  tests/test_compressor.py
  tests/test_carcheck.py
  runs/vae/<YYYYMMDD-HHMM>-<name>/   Runs (gitignored; runs/ already is)
  checkpoints/<name>/                promoted Checkpoints (committed)
  ```
- `pyproject.toml`:
  - `[tool.uv.build-backend] module-name = ["toycar", "wam"]`.
  - `[dependency-groups] model = ["torch>=2.15,<2.16"]` (exact version in `uv.lock`), and `[tool.uv] default-groups = ["dev", "model"]`, so plain `uv sync` / `uv run` include torch. `uv sync --no-group model` gives a torch-free environment.
  - matplotlib stays in the optional `viz` group; `show` and the contact sheet import it lazily.
- `.gitignore`: keep `*.pt`, add the exception `!checkpoints/**`.
- **`toycar` never imports torch**, enforced by a test (§10, test 1).
- Later Steps add modules to `wam` by concern (`model.py`, `train.py`, `sample.py`, …), not one per Step.

---

## 2. Environment

([ticket 01](issues/01-torch-on-mac.md))

- Apple M1 Pro, 16 GB, Python 3.14 (pinned). torch 2.15 has a cp314 macOS arm64 wheel on PyPI with MPS built in.
- Device: **MPS** when available, else CPU (`--cpu` forces CPU). MPS is about 11× faster than CPU for this network.
- **float32, eager.** No autocast / mixed precision (no gain on M1; risks the nose detail), no `torch.compile` (slower in float32 here).
- Data loading in the main process (`num_workers=0`); uint8 batches move to the device and are converted there.
- MPS gotchas: no float64 (go via `.cpu()`), timings need `torch.mps.synchronize()`.
- Long runs: start with `caffeinate -i uv run …` so the Mac does not sleep.

---

## 3. Network and loss

([ticket 03](issues/03-architecture-and-objective.md))

### 3.1 Network

VAE, 64×64×3 → 8×8×(16 means + 16 log-variances) → 64×64×3. **3,166,435 parameters** at the default `arch`.

- `Res(c)` = (GroupNorm, SiLU, 3×3 conv c → c) × 2, plus the skip `x + …`.
- GroupNorm groups = `min(32, c // 4)`. All 3×3 convs use padding 1. No attention. No transposed convs: upsampling is nearest-neighbour ×2 followed by a 3×3 conv.

| Encoder | Output | Decoder | Output |
|---|---|---|---|
| 3×3 conv 3 → c₀ (no Res at 64×64) | 64×64×c₀ | 3×3 conv Z → c₃, `blocks` × Res(c₃) | 8×8×c₃ |
| 3×3 conv stride 2 → c₁, `blocks` × Res(c₁) | 32×32×c₁ | up ×2, 3×3 conv → c₂, `blocks` × Res(c₂) | 16×16×c₂ |
| 3×3 conv stride 2 → c₂, `blocks` × Res(c₂) | 16×16×c₂ | up ×2, 3×3 conv → c₁, `blocks` × Res(c₁) | 32×32×c₁ |
| 3×3 conv stride 2 → c₃, `blocks` × Res(c₃) | 8×8×c₃ | up ×2, 3×3 conv → c₀ | 64×64×c₀ |
| GroupNorm, SiLU, 1×1 conv → 2Z | 8×8×2Z | GroupNorm, SiLU, 3×3 conv → 3 | 64×64×3 |

- `arch` = `{"channels": [32, 64, 128, 128], "blocks": 2, "z": 16}` by default (c₀ … c₃, Z). Tests use a tiny `arch`, e.g. `{"channels": [8, 8, 16, 16], "blocks": 1, "z": 16}`. Z = 16 and the 8×8 grid are fixed by the contract (§7); only channels and blocks vary.
- Encoder output splits into μ (first Z channels) and log σ² (last Z), log σ² clamped to [−30, 20].
- Input `x = frame / 127.5 − 1` in CHW. The decoder output is unbounded; it is clamped to [−1, 1] only when converting to uint8: `round((x + 1) · 127.5)`.

### 3.2 Loss

- **Car region:** pixels where the Frame differs from the empty-track render in the same episode's colours (`render(State(-100, -100, 0, 0), track, rp)`), grown by 1 pixel in all 8 directions. A training-only, privileged mask; `encode` / `decode` never need it.
- **Reconstruction:** weighted mean L1 over pixel values in [−1, 1], weight `w = 1 + λ` inside the car region and 1 elsewhere, **λ = 30**: `rec = Σ w·|x̂ − x| / (3 · Σ w)` (w per pixel, summed over the 3 channels).
- **KL:** `kl = mean over all latent values of 0.5 · (μ² + σ² − 1 − log σ²)`, weight **1e-3**: `loss = rec + 1e-3 · kl`.
- In training the decoder receives `z = μ + σ · ε` (ε from the VAE-noise stream, §5.4). Everywhere after training only **μ** is used.
- No perceptual or adversarial terms.

### 3.3 Latent normalisation

- After training, per-channel **mean and std** of μ over **65,536 train Frames** (drawn from a fixed stream, §5.4), 16 + 16 float32 numbers, stored in the Checkpoint.
- `encode` returns `(μ − mean) / std`; `decode` un-normalises first (`μ = z · std + mean`).
- Near-unused channels are allowed (the prototype had two with std ≈ 0.1). They are only reported (per-channel stats in `metrics.json` and the card), with no rule.

---

## 4. Car-visibility check (`wam/carcheck.py`)

([ticket 04](issues/04-car-visibility-check.md), including its 2026-11-04 amendment)

Compares a reconstruction with its original Frame. It knows nothing about the compressor, so Steps 8 and 10 can also score predicted or Probe Frames with it.

### 4.1 Pose estimator

`estimate(frame, background, rp, centre=None) -> (x, y, θ | None) | None` (metres, radians; None = no car):
- `PX = 64 / 40` pixels per metre. Pixel centres at `(col + 0.5, row + 0.5)`.
- Evidence `diff = Σ_RGB |frame − background|` (background = empty-track render in the episode's colours). Weight `w = clip(diff / 60, 0, 1)`, and `w = 0` where `diff < 9`.
- With `centre` (metres), pixels more than `4 m · PX` from it in x or in y get w = 0.
- No car if `Σ w < 1`. Position = w-weighted centroid.
- Nose pixels = car pixels closer (Euclidean RGB) to `rp.nose` than to `rp.body`, weight `w`. If their weight sum < 0.2, θ is None. Otherwise θ = `atan2(nose centroid − car centroid)` (y down, matching Track units).
- On the 4096 original val Frames vs the true State: position median 0.05 / p95 0.16 / max 0.33 m; heading median 2.4° / p95 9.2° / max 31°.

### 4.2 Evaluation Frames

`eval_frames(es, n=4096) -> list[(episode, tick)]`: episode i of E gets `n // E + (1 if i < n % E else 0)` ticks, drawn without replacement from 0 … length with `default_rng([seed_i, 5])` and sorted. The check uses the **4096 val Frames**; the test Split stays untouched.

### 4.3 Per-Frame measures

For each Frame, with `po = estimate(original, bg, rp)` and `pr = estimate(reconstruction, bg, rp, centre=po[:2])`:
- `found` = `pr is not None`.
- `pos` = distance between `pr` and `po` positions (∞ if not found).
- `head` = absolute wrapped angle between `pr` and `po` headings in degrees. 180 if not found or no nose.
- `readable` = the estimate on the **original** is within **20°** of the true heading. Unreadable Frames are left out of every statistic.
- Diagnostics: PSNR (dB, 0–255) of the whole Frame, of the car region (§3.2) and of the background (the rest; 99 dB if identical); position and heading error of `pr` vs the true State.
- Situations: `section` (`branch != −1`), `fast` (v ≥ 8 m/s), `kerb` (≥ 0.5 m from both Route centerlines).

### 4.4 Verdict

Per slice (`all`, `section`, `fast`, `kerb`), over its readable Frames: `n`, `skip` (unreadable, left out), found share, position p95 / max, heading p95 / max, mean PSNR whole / car / background.

**PASS** on `all` iff **car found in 100%**, **position p95 ≤ 0.3 m and max ≤ 0.6 m**, **heading p95 ≤ 10° and max ≤ 30°**, and `skip ≤ 1%` of the Frames. More than 1% skipped gives **INVALID** (the check itself failed), otherwise FAIL. Slice verdicts are printed as diagnostics; only `all` gates.

Run on:
- **clean** latents (gates);
- **noisy** latents, normalised latents + 0.1·N(0, 1) from a fixed generator (reported; expected to pass);
- each **colour Probe** (`brown_grass`, `light_road`, `blue_car`, `no_kerb`), report only.

### 4.5 Contact sheet

One PNG (`contact.png`), about 30 entries in rows of 5, each *original | reconstruction | 12×12-pixel zoom of each around the car*, with the estimated heading arrows on the zooms and "Δp … m Δθ …°" (red above the max limits). Entries, picked with `default_rng(0)`: 8 random, 4 worst by position, 4 worst by heading (among readable Frames), 2 per situation, 2 per colour Probe.

### 4.6 API

```python
estimate(frame, background, rp, centre=None) -> tuple[float, float, float | None] | None
readable(po, theta) -> bool
eval_frames(es, n=4096) -> list[tuple[int, int]]
check(originals, reconstructions, states, render_params, track, *, branch, label=None) -> dict
    # arrays aligned per Frame; returns {"all"|"section"|"fast"|"kerb": stats, ...} plus per-Frame rows; prints the table when label is given
contact_sheet(path, title, clean, probes) -> None   # needs matplotlib
THRESH = {"pos_p95": 0.3, "pos_max": 0.6, "head_p95": 10.0, "head_max": 30.0}; READABLE_DEG = 20.0; MAX_UNREADABLE = 0.01
```

---

## 5. Training recipe

([ticket 05](issues/05-training-recipe.md); the recipe is prototype run E, which passed — ticket 03)

### 5.1 Recipe

| | |
|---|---|
| Data | each Frame drawn **uniformly over all train Frames** (`frame_dataset(train)`, 1.445 M items, with replacement), rendered on the fly, with its car mask (§3.2) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | 200 linear warm-up steps (`lr · (s + 1) / 200`), then cosine to 0.05× at the last step: `0.05 + 0.95 · ½ (1 + cos(π (s − 200) / (steps − 200)))` |
| Batch / steps | 128 / **12,000** (1.54 M Frames ≈ 1.06 × train). A different `--steps` keeps 200 warm-up steps and stretches the cosine |
| Precision | float32 eager; no compile, no EMA |
| Expected time | ≈ 89 min training (≈ 290 Frames/s on MPS) + ≈ 5 min stats and check ≈ **1 h 35 min** |

Val Frames are for monitoring and the check only; Probes for reporting only. No oversampling of rare situations.

### 5.2 Monitoring and stopping

- **No early stopping:** always the full schedule.
- Every 1,000 steps: `val_rec` = the reconstruction loss (§3.2, z = μ) on 1,024 fixed val Frames (`eval_frames(val, 1024)`).

### 5.3 End of training and freezing

1. Latent normalisation stats (§3.3).
2. The full check (§4) on the 4096 val Frames: clean, noisy, per Probe.
3. The contact sheet.
4. The Run directory (§6.1) is completed.

**Freeze = promote** (§6.3). After promotion, weights and stats never change. Steps 6–10 never fine-tune or jointly train the compressor. A better compressor gets a new name (`vae-v2`) and everything downstream reruns.

A **failed** clean check is not retried with other seeds: the recipe is not robust, so ticket 03's design is reopened (EMA or extra nose weight first). Only runs broken by a crash, NaN or interruption are rerun.

### 5.4 Seeds

One `seed` (default 0) in the config, with separate derived streams:
- weight init: `torch.manual_seed(seed)` before building the network;
- training Frames: `np.random.default_rng([seed, 0])`;
- latent-stats Frames: `np.random.default_rng([seed, 1])`;
- VAE noise ε: a `torch.Generator` on the training device seeded with `seed`;
- the noisy-latent check: a CPU `torch.Generator` seeded with 7.

Same seed + device + torch version + code → a **bit-identical** run. Otherwise **statistically equivalent** (same verdict, close numbers), which is why the Checkpoint is committed.

### 5.5 Resume

- Every 1,000 steps the Run directory gets `resume.pt`: model, optimiser, scheduler, step, the numpy Frame-sampler state and the torch noise-generator state (and the running log position).
- `train --resume runs/vae/<run>` continues from it (losing ≤ ~7 min), using the Run's own `config.json`.
- On the same device and torch, a resumed run is aimed to be bit-identical to an uninterrupted one (tested on CPU, §10 test 10).
- `config.json` records each resume and its step. `resume.pt` is deleted when the run finishes. Resumed runs are promotable.

---

## 6. Runs and Checkpoints

([ticket 02](issues/02-package-layout-and-checkpoints.md))

### 6.1 Run directory

`runs/vae/<YYYYMMDD-HHMM>-<name>/`, gitignored, never overwritten. `<name>` from `--name`, default `vae`. If the directory exists, a `-2`, `-3`, … suffix is added.

| File | Contents |
|---|---|
| `config.json` | training config (arch, λ, KL weight, optimiser, schedule, steps, batch, seed) + provenance (§6.2) + resumes |
| `log.csv` | every 100 steps: `step, loss, rec, kl, lr, frames_per_s, elapsed_s`; every 1,000 also `val_rec` (empty otherwise) |
| `metrics.json` | the full check (clean, noisy, per Probe; per slice incl. `skip`), per-channel latent mean / std, training seconds, params, Frames/s |
| `contact.png` | the contact sheet (§4.5) |
| `checkpoint.pt` | final weights, same layout as a Checkpoint's `model.pt` |
| `stdout.txt` | a copy of everything printed |
| `resume.pt` | only while the run is unfinished |

Terminal progress line every 100 steps: step, loss, rec, KL, lr, Frames/s, ETA.

### 6.2 Checkpoint files

`checkpoints/<name>/`, **committed** (≈ 13 MB; under GitHub's 50 MB warning, no LFS):

- **`model.pt`**: `torch.save` of a plain dict `{"state_dict", "arch", "latent_mean", "latent_std"}` (tensors and plain Python only), loaded with `torch.load(..., weights_only=True)`.
- **`card.json`**:
  - `name`, `created` (ISO date-time), `source_run` (repo-relative);
  - `git_commit`, `git_dirty`, `code_sha256` (sha256 of `src/wam/compressor.py`), `torch_version`, `device`;
  - `dataset` (repo-relative path, e.g. `data/step3-v1`), `manifest_sha256` (sha256 of its `manifest.json`), `split` (`train`), `stats_frames` (65,536);
  - `config` (the full training config) and final losses (`loss`, `rec`, `kl`, `val_rec`);
  - `check`: the summary of §4.4 for clean, noisy and each Probe, with verdicts; per-channel latent std.
- **Only repo-relative paths**; nothing machine-specific beyond commit and device type (the repo will go to GitHub). The same fields, minus `name`, are in the Run's `config.json` / `metrics.json`.

### 6.3 Promote and load

- `promote <run> --name <name>` copies `checkpoint.pt` → `checkpoints/<name>/model.pt` and writes `card.json`. It **refuses** if the name exists (never overwrite), and if the Run's clean check is not PASS unless `--force`.
- `load(name, device=None) -> Compressor` reads `checkpoints/<name>/`:
  - **error** if `arch` does not build or the state dict does not fit;
  - **warning** (Python `warnings.warn`) if `code_sha256` differs from the current `compressor.py`;
  - no dataset check (encoding needs no dataset).
- Later Steps refer to the compressor **by name only**, and store that name in their own configs and cards.

---

## 7. Latent contract for Steps 6–10

([ticket 06](issues/06-latents-for-windows.md))

- **Encoded on the fly, no latent cache.** One path for training (Steps 6–7), Probes (Step 10) and closed loop (Step 9). Measured with the prototype E encoder on MPS: 6.1 ms per Window (17 Frames) plus 2.3 ms Step 3 load + render, ≈ 8–11% of a Step 6 batch-8 step. A cache could later be added behind the same API if Step 7 turns out data-bound.
- A Frame's Latent: float32 **`(16, 8, 8)`**, channel-first, **normalised** (per-channel mean 0, std 1 over train), the **mean** latent (never sampled after training).
- **Token order** for Step 6: `z.flatten(-2).transpose(-1, -2)` → `(…, 64, 16)`, **row-major**; token i is cell (row `i // 8`, col `i % 8`), covering pixels `[8·row, 8·row + 8) × [8·col, 8·col + 8)`, i.e. 5 × 5 m of Track.
- Windows: `encode_windows(batch)` → `obs (B, 16, 8, 8)`, `future (B, H, 16, 8, 8)`; future latent k ↔ action k ↔ future Frame k (Step 3 §6.1).
- Encoding runs in the main process on the compressor's device, after the batch is assembled.
- **Exactness:** the same call on the same input is bit-identical. Across batch sizes (one closed-loop Frame vs a training batch) results agree only within ≈ 1e-5 in normalised units; nobody relies on bit-equality across batch sizes.

---

## 8. API (`wam.compressor`)

```python
ARCH: dict                     # default arch (§3.1)
class VAE(torch.nn.Module):    # the network (§3.1)
    def __init__(self, arch: dict = ARCH) -> None
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]   # x float (B, 3, 64, 64) in [−1, 1] → μ, log σ² (B, Z, 8, 8)
    def decode(self, z: Tensor) -> Tensor                  # raw (unnormalised) latents → (B, 3, 64, 64), unbounded

class Compressor:              # frozen: VAE in eval mode, requires_grad=False on all weights
    name: str | None; card: dict | None; latent_shape: tuple[int, int, int]  # (16, 8, 8)
    device: torch.device; latent_mean: Tensor; latent_std: Tensor
    def encode(self, frames) -> Tensor
        # uint8 (..., 64, 64, 3), torch or numpy, any leading dims → float32 (..., 16, 8, 8) normalised μ,
        # on self.device, under torch.inference_mode()
    def decode(self, z: Tensor, *, as_float: bool = False) -> Tensor
        # normalised (..., 16, 8, 8) → uint8 (..., 64, 64, 3) under inference_mode;
        # as_float=True → float32 (..., 3, 64, 64) in [−1, 1] (clamped), WITHOUT inference_mode: differentiable in z
    def encode_windows(self, batch: dict) -> tuple[Tensor, Tensor]
        # Step 3 Window batch (obs_frame (B, 64, 64, 3), future_frames (B, H, 64, 64, 3)) → (B, 16, 8, 8), (B, H, 16, 8, 8)
        # in one encoder call

def load(name: str, device: str | torch.device | None = None, *, root: Path = Path("checkpoints")) -> Compressor
def promote(run: Path, name: str, *, force: bool = False, root: Path = Path("checkpoints")) -> Path
def train(config: dict, out: Path, *, resume: bool = False, device=None) -> Path   # returns the Run directory
def main(argv: Sequence[str] | None = None) -> int
```

`device=None` picks MPS if available, else CPU. `load` of a Run directory (for `eval` / `show`) reads its `checkpoint.pt` and `config.json` with `name = None`.

---

## 9. CLI

```
python -m wam.compressor train   [--name NAME] [--steps 12000] [--seed 0] [--data data/step3-v1] [--cpu]
python -m wam.compressor train   --resume runs/vae/<run>
python -m wam.compressor eval    <run-dir | checkpoint-name> [--eval-frames 4096] [--json OUT]
python -m wam.compressor show    <run-dir | checkpoint-name> [--out contact.png]
python -m wam.compressor promote runs/vae/<run> --name vae-v1 [--force]
```

- `train`: runs §5 and ends with `eval` and `show` into the Run directory; prints the Run path. Exit 0 if the clean check passes, else 1 (the Run is kept either way).
- `eval`: prints the §4 tables (clean, noisy, per situation, per Probe). Exit **0 on clean PASS, 1 on FAIL or INVALID**. `--json` writes the summary.
- `show`: writes the contact sheet (default: into the Run directory, or `contact.png` for a Checkpoint name). Needs `--group viz`; without matplotlib it prints how to install it and exits 2.
- `promote`: §6.3. Exit 1 on refusal, with the reason.
- An argument is a Run directory if it is an existing path, else a Checkpoint name.
- Hidden options for tests and smoke runs: `--eval-frames`, `--stats-frames`, `--batch`, `--arch` (JSON).

---

## 10. Automated tests (pytest, CPU only)

Fast (< 60 s added), no full training. Most tests use the tiny `arch` (§3.1) and the Step 3 tiny dataset fixture.

`tests/test_compressor.py`:
1. A subprocess importing every `toycar` module leaves `"torch"` out of `sys.modules`.
2. Encoder → `(B, 2Z, 8, 8)`, decoder → `(B, 3, 64, 64)`; the default `arch` has exactly 3,166,435 parameters.
3. `encode` / `decode` dtypes and shapes; any leading dims (e.g. `(2, 17, 64, 64, 3)`); numpy and torch inputs; output on the compressor's device.
4. `encode` equals `(μ − mean) / std`; `decode(encode(x))` equals the raw decoder applied to μ.
5. A repeated `encode` is bit-identical; one Frame alone vs inside a batch agree within 1e-5.
6. `encode_windows` shapes, and equality with encoding `obs_frame` and `future_frames` separately; future k at index k.
7. After `flatten(-2).transpose(-1, -2)`, token i is cell (`i // 8`, `i % 8`).
8. `decode(as_float=True)` gives gradients to `z` and none to the weights.
9. Same seed → equal weights after a few steps; a different seed → different weights.
10. 20 steps straight == 10 steps + resume + 10 steps, bit for bit.
11. Promote / load round trip; refuses an existing name; refuses a failing Run without `--force`; `arch` mismatch → error; code-hash mismatch → warning; no absolute paths in the card.
12. CLI smoke: `train --steps 5` (tiny arch, tiny dataset, small `--eval-frames` / `--stats-frames`) → `eval` exits 1 → `promote --force` → `show` (skipped without matplotlib).

`tests/test_carcheck.py`:
13. Estimator on 256 original val Frames vs the true State: position p95 ≤ 0.2 m, heading p95 ≤ 10°.
14. The six ticket 04 fakes keep their verdicts on 256 Frames: exact copy and background noise ±4 PASS; car shifted 1 px, erased, facing backwards, and 8×8 block average FAIL.
15. Synthetic rows: `skip` is counted, unreadable rows are left out, and > 1% unreadable gives INVALID, not PASS.

Also:
- The Step 3 `test_dataloader_compat` now runs (torch is in the default groups).
- A real-Checkpoint test, skipped if `checkpoints/vae-v1` or `data/step3-v1` is absent: `load("vae-v1")` PASSes the check on 512 val Frames.

---

## 11. Determinism and performance targets

- Training: ≈ 290 Frames/s on MPS at the default arch and batch 128; ≈ 1 h 35 min end to end. Memory well under the 11.8 GB MPS ceiling (≈ 3–5 GB at batch 128).
- Encoding: ≈ 6 ms per Window (17 Frames) on MPS, batch 8 Windows.
- Determinism as in §5.4 and §7.

---

## 12. Manual acceptance (PLAN.md Step 4 check)

([ticket 07](issues/07-api-cli-tests.md))

1. `uv sync` and `uv run pytest`: all pass.
2. `caffeinate -i uv run python -m wam.compressor train --name vae-v1`: about 1 h 35 min, ends with a clean **PASS**. Prototype reference (run E): position p95 0.12 / max 0.26 m, heading p95 8.0° / max 24.7°, car PSNR 38.3 dB. Expect close, not identical, numbers.
3. Open `runs/vae/<run>/contact.png`: in every zoomed pair the red body and yellow nose sit in the same place and point the same way, and the heading arrows agree. This is PLAN's "reconstructions keep the car's position and heading clearly visible".
4. `uv run python -m wam.compressor promote runs/vae/<run> --name vae-v1`, and commit `checkpoints/vae-v1/`.
5. `uv run python -m wam.compressor eval vae-v1`: same verdict, exit 0.
6. `checkpoints/vae-v1/card.json` names the commit, the dataset manifest hash and the PASS, and contains no absolute paths.

Not blocking: the off-Route weakness (prototype heading p95 ≈ 17° on the `kerb` slice) and the colour-Probe failures. Both are recorded on the card and the contact sheet. A failed step 2 follows §5.3.

---

## 13. Hand-offs to later steps

- **Step 6:** `load("vae-v1")`, `encode_windows`, token order (§7). 64 Latent tokens per Frame, about 1100 per Window. Latents are already unit-variance per channel.
- **Step 7:** flow matching on normalised Latents directly; no extra scale factor. If training turns out data-bound, a latent cache can be added behind `encode` (§7).
- **Step 8:** decode predicted Latents with `decode` and score them with `wam.carcheck.check` against the true future Frames. The decoder tolerates normalised noise of 0.1 (§4.4 noisy check).
- **Step 9:** closed-loop Frames from the env's `render` go through the same `encode`, usually one Frame at a time (§7 exactness).
- **Step 10:**
  - The colour Probes already degrade heavily *at the compressor* (it repaints training colours; prototype run E: `brown_grass`, `light_road`, `blue_car` car PSNR 15–24 dB and heading p95 35–140°; `no_kerb` milder at 32 dB and 16°). A policy failure on a Probe may start here.
  - Off-Route Frames are the compressor's weakest situation (thin recovery data, Step 3 §13).
- **A future `vae-v2`:** try EMA weights first, then an extra nose weight in the loss.

---

## 14. Clarifications made while compiling

These fill gaps between ticket decisions. They don't change any decision. If one looks wrong, reopen the relevant ticket.

1. **Car masks need the episode of a sampled Frame.** `FrameItems` returns only the image, so this spec adds one small, additive Step 3 method, `FrameItems.locate(i) -> (episode, tick)`, using the existing offsets. The trainer renders the Frame via `frame_dataset(train)[i]` and the background with `es.render_params(episode)`, cached per episode. No Step 3 behaviour changes.
2. **`arch`** is a dict of channels, blocks and z, the only thing the network is built from, so tests can use a tiny one and `load` can validate it.
3. **Seed streams** (§5.4) are spelt out: Step 3 style `[seed, k]` for numpy, plain seeds for torch.
4. **`val_rec`** uses `eval_frames(val, 1024)` and z = μ, so it is comparable across runs.
5. **KL and reconstruction formulas** are written out from the prototype (§3.2), which is what passed.
6. **Run directory name clashes** get a numeric suffix rather than an error.
7. **The code hash covers `compressor.py` only.** Changes to `carcheck.py` don't affect the weights, so they shouldn't warn on `load`.
8. **`show` without matplotlib** exits 2 with a hint, like a missing optional dependency rather than a failure.
9. **`train` exit code** follows the clean verdict, so scripts can chain `train && promote`.
10. **`--arch` (hidden)** exists so the CLI smoke test can train the tiny network.
11. **Worst-by-heading picks on the contact sheet** are drawn among readable Frames, matching the amended statistic.
12. **The memory target** (§11) is from the ticket 01 measurement scaled to the default arch, not separately decided.

## 15. Out of scope for Step 4

- The Step 6 transformer, Step 7 training and Step 8 sampler; only the latent contract they consume is in scope.
- A latent cache (ticket 06 decided against one; it can be added behind the API later).
- Temporal compression across Frames (one Latent grid per Frame).
- Pretrained or perceptual losses, EMA, mixed precision.
- Changes to Steps 1–3, apart from the additive `FrameItems.locate` (§14.1).
