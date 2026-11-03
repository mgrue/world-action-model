# PyTorch on Python 3.14 and MPS

Ticket: [01-torch-on-mac](../issues/01-torch-on-mac.md). Measured with `prototypes/torch_bench.py` (results in `prototypes/out/torch_bench_*.json`) on the project's Apple M1 Pro (8-core CPU: 6 performance + 2 efficiency, 16 GB), macOS 26.6.2, Python 3.14.6, numpy 2.5.4, torch 2.15.0. Total benchmark runtime about 15 minutes.

## Install

- **torch 2.15.0** (released 2026-10-28) has a `cp314-cp314-macosx_14_0_arm64` wheel on plain PyPI (139 MB), with MPS built in. No extra index, no nightly, no change to the Python pin. Every release since 2.9.0 (2025-10) has a cp314 macOS arm64 wheel.
- Recipe used here, without touching `pyproject.toml` or `uv.lock`:
  ```
  uv run --with torch==2.15.0 python ...
  uv run --with torch==2.15.0 pytest tests/test_dataset.py -k dataloader
  ```
  First install about 12 s. It pulls 9 small pure-Python dependencies: sympy 1.14.0, mpmath 1.3.0, networkx 3.7, filelock 4.2.7, fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4, typing_extensions 4.16.0, setuptools 84.2.1.
- To lock it (ticket 02 decides where), a constraint like `torch>=2.15,<2.16` is enough; uv resolves it from the default index.
- `torch.backends.mps.is_available()` is True. `torch.mps.recommended_max_memory()` is **11.8 GB**, the practical ceiling for model plus activations.
- **`test_dataloader_compat` passes** with torch 2.15.0, and so does the rest of the suite (90 passed).

## Stand-in model

A conv autoencoder 64×64×3 → 8×8×16 → 64×64×3: three stride-2 stages, channels c / 2c / 4c, one GroupNorm + SiLU residual block per stage, nearest upsampling in the decoder. MSE loss, AdamW, input uint8 → [-1, 1] on the device.

| Variant | Params (encoder) | Train GFLOP per Frame | Encoder GFLOP per Frame |
|---|---|---|---|
| small, c = 32 | 1.87 M (0.93 M) | 4.4 | 0.59 |
| large, c = 48 | 4.19 M (2.08 M) | 9.9 | 1.33 |
| lean, c = 32, no residual block at 64×64 | 1.83 M (0.91 M) | 3.5 | 0.44 |

## Training throughput (synthetic uint8 batches already on the device, float32)

| Device | Variant | Batch 64 | Batch 128 | Batch 256 | Peak driver memory (64 / 128 / 256) |
|---|---|---|---|---|---|
| MPS | small | 5.3 steps/s, 338 Frames/s | 2.7, 348 | 1.36, 348 | 2.2 / 3.3 / 5.4 GB |
| MPS | large | 2.7, 172 | 1.4, 176 | 0.69, 176 | 2.3 / 4.4 / 7.7 GB |
| MPS | lean | 7.4, 474 | 3.8, 490 | 1.92, 492 | — / 2.2 / 3.4 GB |
| CPU | small | 0.49, 31 | 0.24, 30 | — | |
| CPU | large | 0.28, 18 | — | — | |
| CPU | lean | 0.67, 43 | 0.33, 42 | — | |

- **MPS is about 11× faster than CPU.** It runs at about 1.5–1.7 TFLOP/s; CPU about 0.15.
- **Frames/s does not grow with batch size.** The GPU is already full at batch 64; a bigger batch just takes proportionally longer. Throughput follows FLOPs per Frame, and most FLOPs sit in the 64×64 layers (removing one residual block there gives +40%).
- One epoch over the train Split (1.445 M Frames) takes about **70 min** for small, 50 min for lean, 2.3 h for large. Consecutive Frames are near-duplicates, so the training recipe (ticket 05) probably needs far less than one epoch.
- Memory is about 21 MB per Frame of batch for small (float32). Batch 256 large (7.7 GB) is the only config near the 11.8 GB ceiling.
- The first step costs 0.4–5 s (Metal shader compilation); later steps are steady.

## Precision and `torch.compile` (MPS)

| Config | small b128 | large b128 | lean b128 | Peak memory, small b128 |
|---|---|---|---|---|
| float32 | 348 Frames/s | 176 | 490 | 3.3 GB |
| float16 autocast | 344 | 176 | 496 | 2.2 GB |
| bfloat16 autocast | 296 | 145 | 411 | 2.2 GB |

| `torch.compile` (b256) | small fp32 | large fp32 | lean fp32 | small fp16 | CPU small fp32 b64 |
|---|---|---|---|---|---|
| eager | 348 | 176 | 492 | 345 | 31 |
| compiled | 224 | 137 | 437 | 384 | 31 |
| compile time | 30 s | 33 s | 20 s | 16 s | 34 s |

- **float16 autocast works** (losses finite, same curve) but is **not faster**: the M1 GPU runs fp16 at the same rate as fp32. It does save about a third of the memory.
- **bfloat16 works but is 15–20% slower** on MPS; on CPU it is 3.4× slower (9 vs 31 Frames/s).
- **`torch.compile` works on MPS** (Inductor's Metal backend, no errors), but costs 16–33 s up front and is **slower in float32** (−11% to −36%). Only fp16 + compile beat eager (+11%), and that's not worth the compile time and extra failure surface. No gain on CPU either.

## Data feeding (`frame_dataset(open_dataset("data/step3-v1", check=False).split("train"))`, 1.445 M Frames)

| Mode | Random Frames/s | Startup |
|---|---|---|
| one process, plain loop | 8,186 | — |
| `DataLoader`, batch 256, `num_workers=0` | 8,036 | 0.07 s |
| `num_workers=2` (spawn, persistent) | 15,900 | 2.4 s |
| `num_workers=4` | 29,465 | 3.6 s |
| `num_workers=6` | 41,752 | 5.4 s |

- Batches come out as `torch.uint8` `[B, 64, 64, 3]` with the default collate; no custom collate needed.
- The full dataset reads at 8k Frames/s per process, about half the 15k measured in Step 3 on a smaller test set. Still **16× faster than training consumes** (≤ 500 Frames/s).
- **End to end** (real Frames through a `DataLoader` into an MPS step, batch 256): small 348 Frames/s with 0 workers and 350 with 4; lean 493 vs 493. MPS runs asynchronously, so the CPU renders the next batch while the GPU computes. **Workers buy nothing for Step 4**, and they cost a few seconds of spawn and extra RAM.

## Encoder cost per Window (for ticket 06)

Encoder only, `no_grad`, a Window = 17 Frames (1 observation + 16 future):

| Device | Variant | ms per Window (8 / 32 Windows per call) | Frames/s |
|---|---|---|---|
| MPS fp32 | small | 5.5 / 5.3 | 3,200 |
| MPS fp32 | large | 10.1 / 9.9 | 1,700 |
| MPS fp32 | lean | 3.3 / 3.2 | 5,300 |
| MPS fp16 | small / large / lean | 5.9 / 11.2 / 3.5 (8 Windows) | slightly slower than fp32 |
| CPU fp32 | small / large / lean | 90 / 148 / 77 | 115–230 |

- **On the fly:** the Step 3 stand-in transformer runs batch 8 in about 615 ms; encoding 8 Windows on MPS adds about 43 ms (small), **about 7%**. On CPU it would add 0.7 s per step (more than doubling it).
- **Cached:** encoding all 1.806 M Frames once takes about **9 min** (small) or 6 min (lean) on MPS. As float16 that's 2 KB per Frame, **3.7 GB** for all Splits (3.0 GB train). That fits on disk; it does not fit comfortably in RAM next to training.
- Both are cheap. Speed doesn't decide ticket 06.

## Determinism

Seeded run: small model, 20 AdamW steps, batch 64, fixed inputs; compared every loss and the final sum of all parameters.

- **MPS: bit-identical** across two runs in the same process and across two separate processes. So is CPU.
- `torch.use_deterministic_algorithms(True)` raises nothing for this model on either device, and the result stays identical.
- **MPS vs CPU are not bit-identical:** losses differ by up to 6e-6 after 20 steps (relative 1e-7 at the end). Tests that compare exact numbers must pick one device. CPU is the safe choice for tests; tests that compare against MPS need a tolerance.

## MPS gotchas found

- **No float64 on MPS**: `.double()` on an MPS tensor raises `TypeError`. Move to CPU first (`.cpu().double()`) for checksums or statistics. Latent normalisation statistics (ticket 03/06) should be accumulated in float32 on MPS or in float64 on CPU.
- **No peak-memory counter**: `torch.mps` has `current_allocated_memory()` (tensors alive now, small between steps) and `driver_allocated_memory()` (includes the allocator cache). Peaks must be sampled. Driver memory is the number that matters for the 16 GB machine.
- **Asynchronous execution**: timings need `torch.mps.synchronize()` or a `.item()`. The same asynchrony is why the loader overlaps with the step for free.
- No unsupported ops hit: conv, GroupNorm, SiLU, nearest upsampling, MSE, AdamW, autocast and `torch.compile` all run natively. `PYTORCH_ENABLE_MPS_FALLBACK` was not needed.
- `torch.compile` prints a harmless "Profiler record function … will be ignored" warning.
- CPU torch uses 6 threads (the performance cores).

## Recommendation

- **Install:** `torch==2.15.0` from PyPI on Python 3.14; keep the Python pin. Pin `torch>=2.15,<2.16` wherever ticket 02 puts it, because MPS numerics and compile behaviour change between minor releases.
- **Device:** MPS for training and encoding. CPU only for unit tests and exact-value checks (11× slower).
- **Precision:** float32. fp16 autocast gives no speed on M1 and only saves memory, which Step 4 doesn't need. Avoid bfloat16.
- **`torch.compile`:** off.
- **Batch:** 64–128 (same Frames/s as 256, less memory, more optimizer steps per Frame).
- **Data feeding:** `DataLoader(frame_dataset(es), batch_size=B, sampler=RandomSampler(..., generator=seeded), num_workers=0)`, uint8 batches moved to MPS and converted there. Add workers only if a later model gets 15× faster.
- **Model size:** compute grows with work at 64×64. Keep the full-resolution stage thin (≤ 32 channels, at most one block). The ~2 M "small"/"lean" class gives 350–500 Frames/s, so an epoch takes about 1 h. 4 M+ doubles that (ticket 03/05).
- **Reproducibility:** seeding (`torch.manual_seed` plus a seeded sampler generator) is enough for bit-identical reruns on the same machine and torch version. Record the device and torch version in the Checkpoint's provenance (ticket 02).

**Implications for Steps 6–8:**
- Step 6 is limited by memory (batch about 8 at about 1100 tokens), not FLOPs. There, fp16 autocast's one-third memory saving may be worth testing, unlike Step 4.
- Per-Window encoding is about 5 ms on MPS, about 7% of a Step 6 step. On-the-fly encoding with a frozen encoder is affordable; caching costs about 9 min and 3.7 GB.
- The `frame_dataset`/`Windows` loaders are fast enough without workers.
