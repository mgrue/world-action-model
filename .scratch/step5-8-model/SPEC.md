# Steps 5–8 spec: `wam` world action model — Instruction encoding, multi-stream transformer, flow-matching training, Euler sampler

Buildable spec for PLAN.md Steps 5–8. A small FLUX-style transformer reads one observed Frame (as `vae-v1` Latents), the State and the Instruction, and jointly denoises 8 future Frames and a 16-step action Chunk by flow matching. It is trained on Step 3 Windows, checked, and frozen as the named **Checkpoint** `wam-v1`. This spec also fixes the **interface contract** Steps 9–10 rely on:
- the Policy API that turns one Frame, State and Instruction into an action Chunk
- the training switches Step 10's experiments need (regression baseline, no video, classifier-free guidance, Chunk length)
- the Checkpoint format

Sources: every decision here was made in a ticket of `MAP.md`, and each section links to its ticket. Vocabulary follows `CONTEXT.md` at the repo root (Frame, State, Instruction, Window, Latent, Token, Stream, Chunk, Null Instruction, Noise level, Checkpoint, Run, …). The contracts this builds on are `.scratch/step3-dataset/SPEC.md` (Windows) and `.scratch/step4-vae/SPEC.md` (`vae-v1`, Run/Checkpoint pattern). The reference implementation whose ideas are mirrored, shrunk, is `/Users/grue/code/flux-action` (read it; don't import it). Measurements and prototypes are in `research/01-throughput-budget.md` and `prototypes/` with their outputs in `prototypes/out/`. Section 16 lists clarifications made while compiling.

PLAN.md checks:
- **Step 7:** "training loss falls for both streams; logged separately." Measured by the Step 7 loss check (§9.1).
- **Step 8:** "sampled action chunks look like the expert's on held-out windows." Measured by the action-error and Fork gates (§9.2–9.3), and seen on the contact sheet (§9.5).

---

## 1. Scope and layout

([ticket 08](issues/08-api-cli-tests.md))

```
src/wam/runs.py        NEW: Run/provenance helpers extracted from compressor.py (_git_info, _sha256, _rel, _now,
                       resume-file handling); compressor.py imports them — behaviour unchanged, guarded by Step 4 tests
src/wam/model.py       ModelConfig (arch + every switch + action-normalisation constants, validated), position ids,
                       patchify / unpatchify, RoPE, Block, LastLayer, WAM network
src/wam/flow.py        action (de)normalisation, t sampling + shift, noising, losses (flow / regression),
                       Euler sampler + CFG — pure maths, no I/O
src/wam/policy.py      Policy, load, act, sample, promote; the CLI (python -m wam.policy)
src/wam/train.py       training loop: Window sampler, input pipeline, EMA, val, logging, resume
src/wam/evaluate.py    eval subsets, Fork references, metrics, gates, contact sheet
tests/test_wam_{model,flow,policy,train,evaluate}.py
runs/wam/<YYYYMMDD-HHMM>-<name>/     Runs (gitignored)
checkpoints/wam-v1[-variant]/        promoted Checkpoints (committed; variants only when cited, §12)
```

- `pyproject.toml`: add `einops` to the `model` dependency group (pure Python; the patchify and attention reshapes read like the FLUX reference).
- `toycar` stays torch-free. No changes to Steps 1–4 apart from extracting `runs.py`.

---

## 2. Environment and budget

([ticket 01](issues/01-throughput-budget.md))

- Apple M1 Pro, 16 GB (MPS ceiling 11.8 GB), Python 3.14, torch 2.15. Measurements in `research/01-throughput-budget.md`.
- **float32, eager, `F.scaled_dot_product_attention`.** On MPS, SDPA is plain attention: no memory-efficient kernel, O(N²) memory, so **the token count is the lever.** bf16 autocast is 20–30% *slower*. No `torch.compile`.
- **The cost is about proportional to params × tokens** (compute-bound, about 1 TFLOP/s effective). Windows/s is flat above batch 16.
- The `vae-v1` encode shares the GPU and **adds serially**: 4.4 ms per Window with 8 future Frames, 2.6 ms for one Frame.
- **Envelope:**
  - 160–300 tokens per Window (2×2 patchify)
  - 8–20 M params
  - batch 32
  - a full Run of about 3–8 h; this spec lands at 162 tokens, 8.2 M params, about 3.4 h

---

## 3. Token layout, positions and Instruction encoding (PLAN Steps 5–6)

([ticket 02](issues/02-tokens-and-positions.md))

### 3.1 Sequence

One Window becomes one fixed sequence of **162 tokens** (defaults: `horizon` H = 16, `future_stride` = 2), concatenated in this order:

| # | Stream | Count | Input → d | t | h, w | Clean / noised | Output head |
|---|---|---|---|---|---|---|---|
| 0–15 | observed Frame | 16 | `frame_in = Linear(64 → d)` (shared) | 0 | grid 0..3 | clean | — |
| 16 | State | 1 | `state_in = Linear(5 → d)` | 0 | 0, 0 | clean | — |
| 17 | Instruction | 1 | `instr = Embedding(4, d)` | 0 | 0, 0 | clean | — |
| 18–145 | future Frame | 8 × 16 = 128 | `frame_in` (shared) | 2, 4, …, 16 | grid 0..3 | noised | `fut_out`: LastLayer d → 64, unpatchify |
| 146–161 | action | 16 | `act_in = Linear(2 → d)` | 1..16 | 0, 0 | noised | `act_out`: LastLayer d → 2 (velocity) |

- **Future Frames:** only the even ones, k = 2, 4, …, H. Step 3 `windows(horizon=H, future_frames=range(stride − 1, H, stride))`, which is 0-based (`(1, 3, …, 15)` by default). All H actions are always present.
  - `future_stride ∈ {1, 2}`, default 2. Stride 1 gives 290 tokens.
  - With `video_stream = false` there are no future-Frame tokens: 34 tokens, `future_frames=()`.
  - `horizon ∈ {4, 8, 16}` gives 54 / 90 / 162 tokens.
- **Clean** means the token is modulated at noise level 0 (§4.2). The concatenation order is irrelevant to the model (RoPE plus full attention) but fixes the slicing.
- **Attention:** full bidirectional over the whole sequence. There is no causal mask, no cross-attention and no padding (sequences are fixed length). As a consequence, nothing can be cached across Euler steps.

### 3.2 Latent tokens: 2×2 patchify

- A normalised Latent (16, 8, 8) from `vae-v1` becomes 16 tokens of 64 values each:
  `rearrange(z, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)`.
  Token i is the 2×2 cell (i // 4, i % 4) of the 8×8 grid and covers 10 × 10 m. This is lossless and is what FLUX does.
- The observed and future Frame tokens share `frame_in`; they differ by t and noise level. The `fut_out` output is unpatchified back to (16, 8, 8).
- **State token:** the Step 3 State features `(x/40, y/40, cos θ, sin θ, v/10)`, as is.

### 3.3 Instruction encoding (PLAN Step 5)

- **Embedding table:** `Embedding(4, d)` with rows `outer` = 0, `inner` = 1, `any` = 2 (Step 3's integer ids) and **`null` = 3**. The row *is* the token; there is no projection. Init is N(0, 0.02).
- **`null` (the Null Instruction) is not `any`.** `any` is real data (the coin-flip Fork). `null` means "Instruction withheld" and never appears in the data.
- **Instruction dropout:** `p_uncond` is the probability per training Window of replacing the Instruction with `null`. **Default 0.1 in every Run** (baselines included), so every Policy can do classifier-free guidance. `p_uncond = 0` is the control. The State is never dropped.
- **The Instruction enters only as a token.** It is not added to the AdaLN modulation vector; FLUX Action's pooled `vector` is likewise zeros. *Documented fallback, not a switch:* if Instruction-following turns out weak, also add the Instruction embedding to the modulation vector.

### 3.4 Positions: 3-axis RoPE

- RoPE over **(t, h, w)**. In each 64-dim head the split is **t 32 / h 16 / w 16**, theta 10000.
- t is counted in **Ticks**: action k and future Frame k share t = k; the observed Frame, State and Instruction are at t = 0.
- h, w is the Latent grid cell for Frame tokens and (0, 0) otherwise.
- **No type embedding.** Per-type projections, t and the noise level separate the Streams; FLUX has none either.
- RoPE is applied to q and k after their RMSNorm. Implementation follows `flux_action/models/positional.py` and `transformer.py` (`rope`, `apply_rope`).

---

## 4. Transformer (PLAN Step 6)

([ticket 03](issues/03-transformer-blocks.md))

### 4.1 Block: FLUX parallel single-stream block

```
h      = (1 + scale) · LN(x) + shift              # LayerNorm without affine, eps 1e-6
q,k,v  = split(W_qkv h)                           # heads of 64
q,k    = RoPE(RMSNorm(q), RMSNorm(k))             # QK RMSNorm (learned scale), eps 1e-6
out    = W_o · SDPA(q, k, v)  +  W_down · (SiLU(W_g h) ⊙ W_u h)     # parallel attention + SwiGLU, hidden 3d
x      = x + gate · out
```
- No biases anywhere in the block. (shift, scale, gate) are per token (§4.2).

### 4.2 Noise conditioning

- **Per-token timestep embedding:** `vec = t_in(sinusoid(1000 · t, 256))`, where `t_in` = Linear(256 → d), SiLU, Linear(d → d), no bias. The sinusoid is FLUX's `timestep_embedding` (cos ‖ sin, max period 10000).
  - Each token uses its own noise level: 0 for the observed Frame, State and Instruction; the Window's t for future Frames and actions.
  - Regression actions use t = 1 (§5.6).
- **One `Modulation` per Stream, shared by all blocks:** 5 of them (obs, state, instr, fut, act), each `Linear(d → 3d, no bias)` applied to `SiLU(vec)` and split into shift, scale, gate. This mirrors FLUX's per-stream, per-stage modulations, and costs 5 × 3d², about 1 M params.
- **Final layer per predicted Stream** (FLUX `LastLayer`): `Linear(LN(x) · (1 + scale) + shift)`, with its own adaLN `Linear(d → 2d, no bias)` on `SiLU(vec)`. The output linear has no bias: d → 64 for future Frames, d → 2 for actions.

### 4.3 Size and initialisation

- **d = 256, 8 blocks, 4 heads × 64, SwiGLU ratio 3.** These are config fields. The head size is fixed at 64 by the RoPE split, so d moves in steps of 64.
  - Exactly **8,229,888 parameters** (prototype count): blocks 6.82 M, Modulations 0.98 M, LastLayers 0.28 M, `t_in` 0.13 M, input projections 0.02 M.
  - Escalation (§6.6): d384×8.
- **adaLN-Zero:**
  - Zero all 5 Modulation linears, so every block starts as the identity.
  - Zero each LastLayer's adaLN linear **and** its output linear, so the initial velocity is 0 and the first loss is about 2.
  - Other linears use `xavier_uniform` (input-projection biases 0). The Instruction Embedding and `t_in` use N(0, 0.02). RMSNorm scales start at 1.
- **No dropout.**
- **Measured:** 236 ms per training step at batch 32 on MPS (model only, 135 Windows/s), about 85 Windows/s end to end with the encode.

| Part | Shape | Params |
|---|---|---|
| frame_in (obs + fut) | Linear(64 → d) + bias | 16.6 k |
| state_in / act_in | Linear(5 → d), Linear(2 → d) + bias | 2.3 k |
| instr | Embedding(4, d) | 1 k |
| t_in | Linear(256 → d), SiLU, Linear(d → d), no bias | 131 k |
| mods × 5 | Linear(d → 3d), no bias | 983 k |
| block × 8 | qkv d → 3d, mlp_in d → 2·768, attn_out d → d, mlp_out 768 → d, RMSNorm q/k | 852 k each |
| fut_out / act_out | LN, ada Linear(d → 2d), Linear(d → 64 / 2) | 279 k |

Prototype: `prototypes/model_proto.py` (out: `model_proto.json`, `model_proto_loss.csv`).

---

## 5. Flow-matching objective (PLAN Step 7)

([ticket 04](issues/04-flow-matching-objective.md))

### 5.1 Action normalisation

Each dimension is standardised with constants measured on train (797,262 Windows × 16), which gives unit variance like the Latents:
```
steer_n    = steer / 0.618                    # mean fixed to 0 (cw / ccw symmetric; measured 0.0016)
throttle_n = (throttle − 0.077) / 0.373
```
The constants live in `ModelConfig` and are saved in every Checkpoint. The Policy takes and returns **raw** actions: it de-normalises and **clips to [−1, 1]**. Measured statistics are in `prototypes/out/action_stats.json`.
- **Steer:** multimodal, with peaks at about ±0.9, ±0.4 and 0; 0.1% at ±1.
- **Throttle:** 81% exactly 0, 10.4% at +1, about 4% near −0.8.
- **Smoothness:** the step-to-step change has std 0.15 per Tick.

### 5.2 Noise level

- **One t per Window**, shared by future Frames and actions (PLAN, FLUX Action).
- `u ~ N(0, 1)`, `t = s·σ(u) / (1 + (s − 1)·σ(u))`, with **`s = t_shift = 1.65`** (e^0.5, FLUX.1's rule at its smallest size).
- The shift tilts training toward high noise, which is where the Branch is decided.

### 5.3 Loss

```
a_n         = normalise(actions)                       (H, 2)
z           = vae-v1 Latents of the future Frames      (H/stride, 16, 8, 8), already unit variance
ε_z, ε_a    ~ N(0, I)
x_z         = t·ε_z + (1 − t)·z ,   x_a = t·ε_a + (1 − t)·a_n
Instruction → null with probability p_uncond
v_z, v_a    = model(obs, State, Instruction, x_z, x_a, t)
L = w_v · mean((v_z − (ε_z − z))²) + w_a · mean((v_a − (ε_a − a_n))²)        # per-Stream means; batch mean
```
- **`w_v = video_loss_weight = 1`, `w_a = action_loss_weight = 1`.** Per-Stream means already stop the 32 action values from drowning in the 8,192 video values.
- There is no per-t weighting.

### 5.4 Logged losses

`loss_video`, `loss_action`, plus `loss_action_hi_t` (t > 0.7) and `loss_action_lo_t` (t ≤ 0.7).

### 5.5 Training switches (validated together by `ModelConfig`)

| Switch | Values (default) | Effect |
|---|---|---|
| `objective` | `flow` / `regression` (`flow`) | §5.6 |
| `video_stream` | bool (true) | false drops the future-Frame Stream entirely: 34 tokens, no future Frames rendered or encoded |
| `video_loss_weight` | float (1) | w_v; 0 is only allowed with `video_stream = false` |
| `action_loss_weight` | float (1) | w_a |
| `t_shift` | float (1.65) | §5.2; also used by the sampler |
| `horizon` | 4 / 8 / 16 (16) | Chunk length H; future Frames at even k ≤ H |
| `future_stride` | 1 / 2 (2) | §3.1 |
| `p_uncond` | float (0.1) | §3.3 |

**Rejected combinations:**
- `regression` with `video_stream = true`
- `video_stream = true` with `video_loss_weight = 0`. That combination would leak: low-t future Frames show the actions' outcome, and the video velocity at sampling would be untrained.
- `horizon` ∉ {4, 8, 16}, `future_stride` ∉ {1, 2}, d not a multiple of 64.

### 5.6 Regression baseline (Step 10)

- **Same model:** zero action-token inputs, noise-level modulation fixed at t = 1, and the action head predicts the clean normalised actions. `L = mean((head − a_n)²)` (times w_a).
- **Inference:** one forward pass.
- **Requires `video_stream = false`.** Step 10 compares it with flow plus `video_stream = false` (isolating the objective), and with full flow as a third data point.

---

## 6. Training recipe (PLAN Step 7)

([ticket 05](issues/05-training-recipe.md))

### 6.1 Recipe

| | Full Run | Quick Run (`--quick`) |
|---|---|---|
| Data | uniform over train Windows (797,262 at H = 16), a seeded permutation per pass; no oversampling, no Probes, test untouched | same |
| Batch / steps | 32 / **32,000** (1.02 M Windows ≈ 1.3 passes) | 32 / **3,200** |
| Optimiser | AdamW lr **3e-4**, betas (0.9, 0.99), weight decay **0.01 on Linear weights only** (not norms, Instruction embedding, biases) | same |
| Schedule | **1,000** linear warm-up steps (`lr · (s + 1) / 1000`), then cosine to 0.05× at the last step: `0.05 + 0.95 · ½ (1 + cos(π (s − 1000) / (steps − 1000)))`; another `--steps` keeps the warm-up and stretches the cosine | same |
| Gradient clip | global norm 1.0 | same |
| EMA | decay `min(0.999, (1 + s) / (10 + s))`, updated every step | same |
| Precision | float32 eager, SDPA, no compile | same |
| Time | ≈ 3.4 h, run under `caffeinate -i` | ≈ 20 min |

- **No early stopping:** always the full schedule.
- **Uniform sampling** gives about 25k `any` Fork Windows per 1 M Windows seen, matches the closed-loop state distribution, and gives regression and flow identical data.

### 6.2 Input pipeline

- Main process, serial: render the Window batch (Step 3), then `vae-v1.encode_windows`, then the step.
  - The encode output is cloned out of inference mode before use in autograd.
  - With `video_stream = false`, Windows come from `future_frames=()` and only the observed Frame is encoded.
- *Documented optimisation, not built:* rendering in DataLoader workers would save about 10% and stay deterministic, since a render is a pure function of the Window index.

### 6.3 Monitoring

| When | What | Data |
|---|---|---|
| every 100 steps → `log.csv` + terminal | `loss, loss_video, loss_action, loss_action_hi_t, loss_action_lo_t, grad_norm, lr, windows_per_s, elapsed_s` (running mean over the 100 steps) | train batches |
| step 0 and every 2,000 steps → `log.csv` | `val_loss_video`, `val_loss_action`, `val_loss_action_hi_t`, `val_loss_action_lo_t` for **raw and EMA** weights | `eval_subset(val, 1024)` at the Run's H; each val Window has a fixed t and fixed noise (seeded), `p_uncond = 0` |
| every 8,000 steps and at the end → `log.csv` | the periodic sampling metrics (§9.4), EMA | periodic set, about 26 s |
| end of Run | the full evaluation (§9) and the contact sheet | §9 |

- Variants leave the video columns empty.
- The terminal progress line every 100 steps shows step, losses, lr, Windows/s and ETA.

### 6.4 Seeds and determinism

- One `seed` (default 0) feeds three streams:
  - init: `torch.manual_seed(seed)` before building the model
  - the Window permutation: numpy `default_rng([seed, 1])`, re-drawn each pass
  - a CPU `torch.Generator` seeded `seed` for t, noise and `p_uncond`, drawn on the CPU and then moved to the device
- Val uses its own CPU generator seeded 0, independent of the Run seed (§16.3).
- Same seed, device, torch and code gives a **bit-identical** Run; otherwise it is statistically equivalent.

### 6.5 Resume

- Every 1,000 steps the Run directory gets `resume.pt`: the model, EMA, optimiser, scheduler, step, the numpy sampler state, the torch generator state and the log position. It is about 6 min of work at risk.
- `train --resume runs/wam/<run>` continues from it using the Run's own `config.json`, which records each resume and its step.
- `resume.pt` is deleted when the Run finishes. Resumed Runs are promotable.

### 6.6 Freezing and failure

- **Promote `wam-v1`** (default switches) only when all of these hold:
  1. the full schedule finished without NaN
  2. the Step 7 loss check passes (§9.1)
  3. every §9 gate passes
- **Every full Run is preceded by a quick Run** that passes the Step 7 check.
- **A failed `wam-v1` check is not re-rolled with another seed.** Only crashes, NaN or interruptions are rerun. Reopen by symptom, each as a new quick Run and then a full Run:

| Symptom | First fix |
|---|---|
| Action loss stalls | `action_loss_weight 5` |
| Underfit (train ≈ val, both plateaued) | d384×8 |
| Fork commit fails | oversample Fork Windows |
| Still improving at the end | 2× steps |

- **Step 10 variants** (`wam-v1-regression`, `-novideo`, `-h8`, …) are experiments. They are evaluated with the same report and promoted with `--force` and their verdicts in the card, never gated.

---

## 7. Sampler (PLAN Step 8)

([ticket 06](issues/06-sampler.md))

```
ts  = timeshift(t_shift, linspace(1, 0, n_steps + 1))         # timeshift(a, t) = a·t / (1 + (a − 1)·t)
x_z, x_a ~ N(0, I)                                            # CPU torch.Generator, then moved to device
for t, t_next in pairs(ts):
    v_z, v_a = guided_model(obs, State, Instruction, x_z, x_a, t)
    x_z, x_a = x_z + (t_next − t)·v_z ,  x_a + (t_next − t)·v_a
actions = clip(denormalise(x_a), −1, 1);  latents = x_z
```
- **Integrator:** plain Euler, **`n_steps = 10`** by default, on the Checkpoint's training `t_shift`. FLUX DROID's UniPC solver (4 steps) is noted as a later speed-up, not built.
- **Classifier-free guidance, per Stream:** `v = v_uc + w·(v_c − v_uc)`, where `v_uc` uses Instruction `null`.
  - Separate **`cfg_action`** and **`cfg_video`**, both **1.0 by default (off, single pass)**.
  - When either is not 1, one batched forward of 2B runs (conditional ‖ null).
  - Rejected for `objective = regression`.
- **Regression:** one forward pass with zero action inputs at t = 1; `n_steps` and CFG don't apply. `n_samples > 1` returns identical copies.
- **Measured on MPS** (`prototypes/sampler_proto.py`):

| Model | B | Steps | CFG | ms per call | ms per Chunk |
|---|---|---|---|---|---|
| video (162 tok) | 1 | 10 | off | 65.9 | 65.9 |
| video | 1 | 10 | 2.0 | 87.5 | 87.5 |
| video | 1 | 4 / 20 | off | 27.0 / 121.7 | |
| video | 256 | 10 | off | 5,500 | 21.5 |
| no video (34 tok) | 1 | 10 | off | 44.9 | 44.9 |
| no video | 256 | 10 | off | 988 | 3.9 |

  A Step 9 replan costs about 2.6 ms of encode plus 66 ms, about 70 ms.

---

## 8. Policy API for Steps 9–10 (`wam.policy`)

([tickets 06](issues/06-sampler.md), [08](issues/08-api-cli-tests.md))

```python
def load(name: str | Path, device: str | torch.device | None = None, *, root: Path = Path("checkpoints")) -> Policy
    # checkpoints/<name>/ (model.pt + card.json) or a Run directory (its checkpoint.pt + config.json);
    # EMA weights; loads the compressor by the name in the card / config (vae-v1); device None → MPS else CPU

class Policy:
    config: ModelConfig; card: dict | None; compressor: Compressor; device: torch.device
    horizon: int; future_k: np.ndarray            # (K,) e.g. [2, 4, …, 16]; empty without video

    def act(self, frame, state, instruction, *, n_samples: int = 1, n_steps: int = 10,
            cfg_action: float = 1.0, cfg_video: float = 1.0, seed: int | torch.Generator | None = None,
            return_latents: bool = False) -> ActResult
        # frame uint8 (64,64,3) or (B,64,64,3); state = State.features() (5,) or (B,5);
        # instruction "outer"/"inner"/"any" or int, or a sequence of B. n_samples repeats inputs along the batch.

    @torch.no_grad()
    def sample(self, obs_z: Tensor, state: Tensor, instr: Tensor, *, n_steps=10, cfg_action=1.0, cfg_video=1.0,
               generator: torch.Generator | None = None) -> tuple[Tensor | None, Tensor]
        # tensor level, batched: (B,16,8,8), (B,5), (B,) long → (future Latents (B,K,16,8,8) normalised | None,
        # actions (B,H,2) normalised). Used by act() and inside training (periodic sampling).

@dataclass
class ActResult:
    actions: np.ndarray            # float32 (B, n_samples, H, 2), raw units, clipped to [−1, 1]
    latents: Tensor | None         # (B, n_samples, K, 16, 8, 8) normalised; policy.compressor.decode → Frames
    future_k: np.ndarray           # (K,)

def promote(run: Path, name: str, *, force: bool = False, root: Path = Path("checkpoints")) -> Path
def train(config: dict, out: Path, *, resume: bool = False, device=None) -> Path          # in wam.train
def main(argv: Sequence[str] | None = None) -> int
```
- **Shapes:** single inputs keep the leading B = 1. Nothing is squeezed.
- **Seeds:** the same seed gives an identical Chunk on the same device.
- **Replanning rate** (how many actions to execute before calling `act` again) is Step 9's choice, an inference argument and not a training switch.

---

## 9. Evaluation and checks (PLAN Steps 7–8)

([tickets 05](issues/05-training-recipe.md), [07](issues/07-evaluation-and-checks.md))

All sampling uses EMA weights, 10 Euler steps, no CFG and fixed seeds.

### 9.1 Step 7 loss check (gate)

For **both** Streams (action only without video): final EMA val loss ≤ **0.5 ×** its step-0 value (about 2.0, from the zero-init) **and** ≤ **1.05 ×** the lowest val loss seen during the Run.

### 9.2 General action error (gate)

- **Set:** `eval_subset(val, 1024)` minus Windows with `approaches_section`, which leaves 951 Windows at H = 16 with a single valid answer (`prototypes/out/action_baselines.json`; an earlier draft said 970).
- **Metric:** **one** sample per Window, RMSE against the clean label in raw units per dimension, over all H steps; also over k ≤ 4 (reported).
- **Gate:** **`rmse_steer ≤ 0.15` and `rmse_throttle ≤ 0.15`.**
- **Baselines on the same Windows** (reported):

| Prediction | RMSE steer | RMSE throttle |
|---|---|---|
| train mean | 0.61 | 0.39 |
| zero | 0.61 | 0.40 |
| shuffled labels | 0.86 | 0.54 |
| labels + N(0, 0.1) | 0.10 | 0.10 |

  The regression variant's numbers are added once it exists. Executed actions equal clean ones on clean-future Windows, so there is no Expert noise floor.

### 9.3 Fork metrics (gates)

- **References** (`evaluate.fork_references(ds)`):
  - Take the train label Chunks of Windows with `approaches_section` (all Instructions) and use **Direction-normalised steer**: steer × (+1 cw, −1 ccw).
  - Group them by **entry offset** e = the first k ∈ 1..16 with `branch[t+k] ≠ −1`.
  - Average per (e, Planned branch): 16 × 2 reference Chunks of 16 values.
- **Classification:** a Chunk goes to the nearer reference (RMS over its steer values) and **commits** if `r = d_near / (d_near + d_far) < 0.3`.
- **Only Windows with e ≤ 12 are used.** Beyond that the Branches haven't separated: label accuracy is 91% at e = 16.
- **End-of-Run Fork set:** 256 `any` + 128 `outer` + 128 `inner` val Windows with e ≤ 12, stratified over e (§16.5), × **16 samples** each, about 3 min.

| Metric | Definition | Gate | Labels | Averaged references | Shuffled labels |
|---|---|---|---|---|---|
| Commit rate | fraction of all Fork samples with r < 0.3 | **≥ 0.90** | 0.9995 | 0.00 | 0.63 |
| Following rate | fraction of `outer` / `inner` samples classified as the instructed Branch | **≥ 0.90** | 1.00 | — | 0.50 |
| `any` balance | fraction of `any` samples classified outer | **0.30–0.70** | 0.50 | — | — |
| `any` two-sided | fraction of `any` Windows whose 16 samples contain both Branches | **≥ 0.80** | — | — | — |

- Fakes validation is in `prototypes/fork_metric.py` (out `fork_metric.json`).
- **Measured on train labels:** the outer Chunk sweeps 0.52 → −0.78 and the inner rises 0.57 → 0.82; separation d′ grows from 0.3 (k = 1) to 11 (k = 16).
- The regression variant is expected to fail the commit gate, which is PLAN's experiment.

### 9.4 Periodic set (report only, every 8,000 steps)

192 general Windows (1 sample each) and 64 Fork Windows (32 `any`, 16 `outer`, 16 `inner`, 16 samples each). Columns `rmse_steer, rmse_throttle, commit, follow, any_balance, any_two_sided` in `log.csv`.

### 9.5 Report only (`metrics.json`)

- **Imagined future** (with video), on 256 general Windows:
  - Decode the predicted Latents with `vae-v1` and run `carcheck.score_frame(true_future, imagined, state_{t+k}, …)`.
  - Report median and p95 position and heading error per k ∈ {2, …, 16}.
  - Floors: the same on the vae-v1 reconstruction of the true future Frame, and on the true Frame itself.
- Best-of-16 RMSE and per-k RMSE curves.
- **With `eval --sweeps`:**
  - CFG w ∈ {1, 1.5, 2, 3}, both / action only / video only: following and commit rate
  - `n_steps ∈ {2, 4, 10, 20}`: general RMSE and commit rate
- **Contact sheet `samples.png`:**
  1. 6 general Windows: the observed Frame, true future Frames at k = 4, 8, 12, 16, and the imagined Frames below them
  2. the same 6: steer and throttle versus k, label against 16 samples
  3. 6 `any` Fork Windows: 16 Direction-normalised steer samples over the two references

### 9.6 Eval data and threshold discipline

- **Eval data is computed, not stored.** References and subsets (`evaluate.subsets(ds)`) are deterministic functions of the dataset and fixed seeds. Their hash goes into `metrics.json` and the card.
- **Thresholds were set before any trained model existed.** They may be **tightened once**, from the first quick Run, before the full `wam-v1` Run, in a commit that edits this spec and cites the numbers. They are **never loosened** after a full-Run result. The card records the thresholds used.

---

## 10. Runs and Checkpoints

([tickets 05](issues/05-training-recipe.md), [08](issues/08-api-cli-tests.md); Step 4 pattern)

### 10.1 Run directory `runs/wam/<YYYYMMDD-HHMM>-<name>/`

| File | Content |
|---|---|
| `config.json` | the full `ModelConfig` (arch, every switch, normalisation constants), recipe (steps, batch, lr, …), seed, data path; provenance (git sha + dirty, `vae-v1` card hash, dataset manifest hash); resumes |
| `log.csv` | §6.3 columns (empty where not logged at that step) |
| `metrics.json` | the §9 gates and report, thresholds used, eval-data hash, verdict |
| `stdout.txt` | terminal output |
| `samples.png` | contact sheet (§9.5), if matplotlib is available |
| `checkpoint.pt` | at the end: **EMA weights** + `ModelConfig`; enough to build a Policy alone |
| `resume.pt` | only while unfinished (§6.5); the only place raw weights live |

### 10.2 Checkpoint `checkpoints/<name>/`

- `model.pt` is the EMA `checkpoint.pt`. At fp32, 8.2 M params is about 33 MB.
- `card.json` holds:
  - name and created time
  - the full config incl. switches and normalisation constants
  - compressor name (`vae-v1`) and its card hash
  - dataset manifest hash
  - git sha and dirty flag
  - the sha256 of each `src/wam/*.py`
  - metrics, thresholds and verdict
  - eval-data hash and resumes
- There are no absolute paths in the card.

### 10.3 Promote and load

- **`promote`** refuses an existing name, an unfinished Run (no `checkpoint.pt` / `metrics.json`), or a failing Run without `--force`.
- **`load`** builds from `ModelConfig` and checks the compressor's card hash (error on mismatch). It **warns** if `model.py` or `flow.py` differ from the card's hashes (§16.6).

---

## 11. CLI

```
python -m wam.policy train   [--name NAME] [--steps 32000 | --quick] [--seed 0] [--data data/step3-v1]
                             [--set key=value ...] [--cpu]
python -m wam.policy train   --resume runs/wam/<run>
python -m wam.policy eval    <run-dir | checkpoint-name> [--json OUT] [--sweeps]
python -m wam.policy show    <run-dir | checkpoint-name> [--out samples.png]
python -m wam.policy promote runs/wam/<run> --name wam-v1[-variant] [--force]
```
- **`train`:**
  - Every switch goes through `--set` (e.g. `--set objective=regression --set video_stream=false`), validated by `ModelConfig`; an invalid combination exits 2.
  - It ends with `eval` and `show` into the Run directory and prints the Run path.
  - Exit 0 iff the Step 7 check and every §9 gate pass, so `train && promote` works.
- **`eval`:** prints the gate table and the report. Exit **0 on PASS, 1 on FAIL**. `--json` writes the summary. `--sweeps` adds §9.5's sweeps (about 25 min).
- **`show`:** writes the contact sheet. It needs `--group viz`; without matplotlib it prints a hint and exits 2.
- **`promote`:** §10.3. Exit 1 on refusal, with the reason.
- An argument is a Run directory if it is an existing path, else a Checkpoint name.
- Hidden options for tests and smoke runs: `--arch` (JSON: d, depth), `--batch`, `--eval-windows`.

---

## 12. Automated tests (pytest, CPU only)

Fast (< 60 s added), no full training. They use the tiny arch (d64 × 2) and the Step 3 tiny fixture (`tests/conftest.py::tiny_data`, whose 60-Tick episodes have no reliable Fork Windows, hence the synthetic Fork tests).

`tests/test_wam_model.py`
1. Position ids equal §3.1's table row for row (162). Counts are 290 / 90 / 54 / 34 for stride 1, H 8, H 4, no video.
2. Patchify followed by unpatchify is exact; token i = cell (i // 4, i % 4), channels `(c ph pw)`.
3. The default arch has exactly 8,229,888 parameters.
4. Zero-init heads output exactly 0.
5. `ModelConfig` rejects: regression with video, w_v = 0 with video, H ∉ {4, 8, 16}, stride ∉ {1, 2}, d % 64 ≠ 0.
6. RoPE is relative: a common shift of every t leaves the outputs unchanged.
7. Permuting the tokens (with their positions and modulation) permutes the outputs.

`tests/test_wam_flow.py`
8. Noising follows x_t = t·ε + (1 − t)·x with target ε − x. The t sampler stays in (0, 1) with median shift(0.5) = 0.6226.
9. Action normalisation round-trips, with the constants of §5.1.
10. The loss is a weighted sum of per-Stream means. No-video has no video term. Regression's target is a_n. `p_uncond` is about 0.1 over 10k draws.
11. Euler with a constant-velocity fake model gives exactly noise − v. The forward count equals `n_steps`. The schedule runs from 1 to 0 with the shift.
12. CFG at w = 1 is one pass; at w ≠ 1 it uses the formula with a 2B batch. Regression with CFG raises.

`tests/test_wam_policy.py`
13. `act` returns (B, n_samples, H, 2) float32 in [−1, 1]. B = 1 is kept. Instruction works as a name or an int. The same seed gives an identical result; a different seed differs. `return_latents` has the right shape. A zero-init model returns exactly clip(denormalise(initial noise)).
14. Promote/load round trip. Promote refuses an existing name, an unfinished Run, and a failing Run without `--force`. A code-hash mismatch warns. The card has no absolute paths.

`tests/test_wam_train.py`
15. The same seed gives equal weights after a few steps; a different seed gives different weights.
16. 20 steps straight equal 10 steps + resume + 10 steps, bit for bit, including the EMA and the optimiser.
17. The EMA update follows its warm-up formula. Weight decay applies to Linear weights only.
18. Val loss is deterministic: two calls give equal values.

`tests/test_wam_evaluate.py`
19. On synthetic Chunks, references and entry offsets: labels pass; the averaged references fail commit; shuffled labels fail following; a collapse to one Branch fails two-sided. The general RMSE of labels is 0.
20. The subsets and their hash are deterministic.
21. CLI smoke: tiny `train --steps 5` → `eval` exits 1 → `promote --force` → `show` (skipped without matplotlib). `--set objective=regression --set video_stream=false` trains 2 steps. An invalid `--set` exits 2.

Conditional:
22. Real data (skipped if `data/step3-v1` is absent): the Fork references and fakes reproduce §9.3. Labels commit ≥ 0.999 and the averaged references commit 0. Arrays only, seconds.
23. Real Checkpoint (skipped if `checkpoints/wam-v1` is absent): `load` and `act` on 8 val Windows give the right shapes and values in [−1, 1].

The existing `toycar` torch-free test and all Step 4 tests stay green, which guards the `runs.py` extraction.

---

## 13. Performance targets

| | Value |
|---|---|
| Training | ≈ 236 ms per step at batch 32 on MPS (model), ≈ 85 Windows/s end to end incl. encode |
| Full Run | 32,000 steps ≈ 3.4 h |
| Quick Run | 3,200 steps ≈ 20 min |
| Memory | ≈ 3–5 GB at batch 32 |
| Sampling | 66 ms per Chunk at B = 1 (≈ 70 ms per Step 9 replan); 21.5 ms per Chunk batched |
| Periodic eval | ≈ 26 s |
| End-of-Run eval | ≈ 4 min without sweeps |
| Variants | no-video and regression are ≈ 5× cheaper per step in the model; the encode dominates |

---

## 14. Manual acceptance (PLAN Steps 7 and 8)

1. `uv sync && uv run pytest`: all green, real-data tests included.
2. **Quick Run:** `caffeinate -i uv run python -m wam.policy train --quick --name quick` (about 20 min). The Step 7 line must read PASS. If thresholds are tightened (§9.6), it happens **now**, once.
3. **Full Run:** `caffeinate -i uv run python -m wam.policy train --name v1` (about 3.4 h), exit 0.
4. **Look at `samples.png`:**
   - the sampled Chunks lie on the labels
   - the `any` Fork panel shows **two bundles**, one along each reference, and not one bundle down the middle
   - the imagined Frames show the car moving along the Route
5. `uv run python -m wam.policy promote runs/wam/<run> --name wam-v1`, then commit `checkpoints/wam-v1/`.
6. `uv run python -m wam.policy eval wam-v1`: the same verdict, exit 0.

- **PLAN Step 7 check:** `log.csv` has separate `loss_video` / `loss_action` columns and their val versions, and eval's Step 7 line passes.
- **PLAN Step 8 check:** eval's gate table passes (§9.2–9.3).
- **Step 10 variants:** promote them with `--force`, and commit one only when a Step 10 result cites it.

---

## 15. Hand-offs to later steps

- **Step 9 (closed loop):**
  - `policy = wam.policy.load("wam-v1")`
  - each replan: `policy.act(env_frame, state.features(), instruction, seed=…)`, then execute the first few actions of `actions[0, 0]`
  - about 70 ms per replan on MPS
  - The replanning rate is Step 9's argument. Frames from the env's `render` go through the same `vae-v1` encode inside `act`.
- **Step 10 experiments,** as switches and arguments:

| Experiment | How |
|---|---|
| Regression baseline | `--set objective=regression --set video_stream=false`, compared with `--set video_stream=false` (flow) and the full model |
| No video | `--set video_stream=false`. Caveat: less compute per action, which is inherent to the question |
| Classifier-free guidance | every Policy is trained with `p_uncond = 0.1`; sweep `cfg_action` / `cfg_video` in `act` or `eval --sweeps`; `p_uncond=0` is the control |
| Chunk length | `--set horizon=8` / `4`. The Fork effect grows with k (d′ 1.6 at k = 8 vs 11 at k = 16), so short Chunks have less of a two-answer problem to show |
| Small world changes | Probes through Step 3's `windows(probe=…)`. The colour Probes already degrade at the compressor (Step 4 §13) |

- **Known weak spots to watch:** off-Route Frames (thin recovery data, `vae-v1`'s weakest situation); Instruction-following (fallback in §3.3); the Fork commit (fix order in §6.6).
- **Possible later improvements, not built:** UniPC sampling (fewer steps), DataLoader-worker rendering (≈ 10%), per-Stream private blocks (PLAN "later"), a pretrained text encoder for free-form Instructions.

---

## 16. Clarifications made while compiling

These fill gaps between ticket decisions without changing any decision. If one looks wrong, reopen the relevant ticket.

1. **Zero future Frames work in Step 3 and `vae-v1` as they are.** Verified: `windows(future_frames=())` gives `(0, 64, 64, 3)` and `encode_windows` returns `(B, 0, 16, 8, 8)`; `windows(horizon=8, future_frames=range(1, 8, 2))` works. No Step 3 change is needed for the variants.
2. **The Instruction ids** reuse Step 3's (`outer` 0, `inner` 1, `any` 2), so `null` is 3.
3. **Val determinism:** the val subset's t and noise come from a CPU generator seeded 0, independent of the Run seed, so val curves are comparable across Runs. Step-0 val is computed before the first optimiser step, when EMA equals raw.
4. **Variant evaluation at H < 16:**
   - The general set is `eval_subset(val, 1024)` from `windows(horizon=H)` minus `approaches_section`.
   - Fork references are truncated to the first H steps, and Windows need e ≤ min(12, H).
   - Variants are never gated, so no threshold changes.
5. **Stratified Fork set:** for each Instruction, the e ≤ 12 val Windows are split by e, and `n // 12` (plus one for the first `n % 12` offsets) are drawn per offset without replacement with numpy `default_rng([7, instruction_id, e])`, then sorted. The periodic set uses the same rule with its own counts and `[8, …]`. The general periodic 192 is the first 192 of a `default_rng(9)` permutation of the 951-Window general set.
6. **Code-hash warning scope:** the card records every `src/wam/*.py` hash, but `load` warns only for `model.py` and `flow.py`, the files that define what the weights mean, as Step 4 restricted its warning to `compressor.py`.
7. **Run directory name clashes** get a numeric suffix rather than an error (Step 4).
8. **Hidden `--arch`, `--batch`, `--eval-windows`** exist for the CLI smoke test and tiny runs (Step 4 pattern).
9. **Inference-mode Latents:** `Compressor.encode` runs under `torch.inference_mode()`, so training clones the Latents before using them in autograd (found in the ticket 03 prototype).
10. **Regression with `n_samples > 1`** returns identical copies, so shapes stay uniform for Step 10 code.
11. **`horizon` and the clean-future filter:** Step 3's filter depends on H, so a `horizon=8` Run trains on its own (larger) Window set; the normalisation constants stay as in §5.1.
12. **The memory target (§13)** is scaled from ticket 01's measurements, not separately decided.

---

## 17. Out of scope

- Step 9's closed-loop evaluation and Step 10's experiments themselves. Only the API and training switches they need are in scope.
- Changing the frozen compressor (that would be `vae-v2` and a new map).
- Free-form text Instructions and pretrained text encoders (PLAN Step 5 "optional later").
- Multi-GPU, cloud training, mixed precision (bf16 measured slower), `torch.compile`, a latent cache.
- Per-Stream private blocks (PLAN "later"), UniPC or other higher-order samplers.
- Changes to Steps 1–4, apart from extracting `wam/runs.py` (§1).
