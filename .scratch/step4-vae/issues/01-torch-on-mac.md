# PyTorch on Python 3.14 and MPS

Labels: wayfinder:research
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

Can we run PyTorch for Steps 4–8 on this machine (M1 Pro, 16 GB, Python 3.14 pinned), and how fast is a small conv autoencoder there?

- Availability: which torch release has macOS arm64 wheels for Python 3.14, installable with `uv`? If none, what are the options (lower the `requires-python` pin for a separate model project, CPU-only, nightly)?
- Throughput on MPS vs CPU for a stand-in 64×64 → 8×8×16 conv autoencoder (about 1–5 M params): train steps/s and Frames/s at a few batch sizes, peak memory, and whether `frame_dataset` (re-rendering, numpy) keeps up with or without `DataLoader` workers (spawn on macOS).
- MPS gotchas that matter here: unsupported ops, float16/bfloat16 support, determinism (can a seeded run be reproduced? is CPU needed for exact tests?), `torch.compile` availability.
- Does the Step 3 `test_dataloader_compat` pass once torch is installed?

Check: a findings file under `research/` with the install recipe (versions, lock), the measured numbers, and a recommendation for device, precision and data-feeding setup; a throwaway benchmark under `prototypes/`.

## Comments
**Resolution (2026-11-03):** Findings in [research/01-torch-on-mac.md](../research/01-torch-on-mac.md), measured with `prototypes/torch_bench.py` on a ~2 M and a ~4 M param stand-in conv autoencoder.
- **Install:** `torch==2.15.0` has a cp314 macOS arm64 wheel on plain PyPI with MPS built in (cp314 wheels exist since 2.9.0). `uv run --with torch==2.15.0 ...` works; the Python 3.14 pin stays. Pin `torch>=2.15,<2.16` wherever ticket 02 puts it. `test_dataloader_compat` passes (whole suite: 90 passed).
- **Speed:** MPS is about 11× faster than CPU. ~2 M params: 350–490 Frames/s on MPS (about 1.5–1.7 TFLOP/s) vs 30–43 on CPU. ~4 M: 176 vs 18. Frames/s is flat from batch 64 to 256, so use batch 64–128. One epoch over the 1.445 M train Frames takes about 1 h at ~2 M params. Most compute sits in the 64×64 layers.
- **Memory:** about 21 MB per Frame of batch (float32, ~2 M params): 3.3 GB at batch 128, 7.7 GB for ~4 M params at batch 256, against an 11.8 GB MPS ceiling.
- **Precision / compile:** float16 autocast works but isn't faster on M1 (saves a third of the memory). bfloat16 is 15–20% slower. `torch.compile` works on MPS but costs 16–33 s and is slower in float32. Use **float32, eager**.
- **Data feeding:** `frame_dataset` reads 8k random Frames/s in one process (16k / 29k / 42k with 2 / 4 / 6 spawned workers). Training needs ≤ 500, and MPS asynchrony overlaps loading with the step. **`num_workers=0`**, uint8 batches converted on the device.
- **Determinism:** seeded runs are bit-identical on MPS (same process and across processes) and on CPU. `use_deterministic_algorithms(True)` works on both. MPS vs CPU differ by about 1e-6, so exact-value tests run on CPU.
- **Gotchas:** no float64 on MPS (`.double()` raises; go via `.cpu()`). No peak-memory counter (sample `driver_allocated_memory()`). Timings need `torch.mps.synchronize()`. No unsupported ops hit.
- **For ticket 06:** encoder cost per Window (17 Frames) on MPS is about 5 ms (~2 M params), 3 ms (lean) or 10 ms (~4 M), about 7% of a Step 6 batch-8 step. Caching all 1.806 M Frames takes about 9 min and 3.7 GB as float16. CPU encoding (75–150 ms per Window) is too slow for on-the-fly use.
