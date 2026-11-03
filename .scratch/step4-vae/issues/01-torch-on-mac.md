# PyTorch on Python 3.14 and MPS

Labels: wayfinder:research
Status: open
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
