# Storage format and loading throughput

Labels: wayfinder:research
Status: open
Assignee:
Blocked by:
Parent: ../MAP.md

## Question

How should episodes be stored so that Step 4 (frame autoencoder) and Steps 6–8 (window training on MPS) can read them fast enough, on a 16 GB Mac, with a few thousand episodes?

- Candidates: per-episode compressed `.npz` (like Step 1 recordings); one big uint8 `np.memmap`/`.npy` per array (frames, states, actions, flags) with an episode index; HDF5/zarr; **no stored Frames at all** — store States + RenderParams and re-render on the fly (renderer is pure and deterministic, ≈ 0.07 ms/Frame).
- Measure: disk size, write time, random-window read throughput (windows/s for 17 Frames + 16 actions) for each candidate, single process and with a PyTorch `DataLoader` with workers on macOS (spawn). Compare against a rough training step budget (a 5–20 M-parameter transformer on MPS).
- What a window costs in memory as uint8 vs float32, and whether the whole dataset fits in RAM.
- Does re-rendering stay bit-identical across processes and runs (Step 1 guarantees same machine + pinned versions)? What happens to Frames if `render.py` changes later (version/hash in metadata)?

Check: a short findings file under `research/` with measured numbers per candidate and a recommended format (with rationale), plus the provenance metadata it must carry.
