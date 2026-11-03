# Storage format and loading throughput

Labels: wayfinder:research
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

How should episodes be stored so that Step 4 (frame autoencoder) and Steps 6–8 (window training on MPS) can read them fast enough, on a 16 GB Mac, with a few thousand episodes?

- Candidates: per-episode compressed `.npz` (like Step 1 recordings); one big uint8 `np.memmap`/`.npy` per array (frames, states, actions, flags) with an episode index; HDF5/zarr; **no stored Frames at all** — store States + RenderParams and re-render on the fly (renderer is pure and deterministic, ≈ 0.07 ms/Frame).
- Measure: disk size, write time, random-window read throughput (windows/s for 17 Frames + 16 actions) for each candidate, single process and with a PyTorch `DataLoader` with workers on macOS (spawn). Compare against a rough training step budget (a 5–20 M-parameter transformer on MPS).
- What a window costs in memory as uint8 vs float32, and whether the whole dataset fits in RAM.
- Does re-rendering stay bit-identical across processes and runs (Step 1 guarantees same machine + pinned versions)? What happens to Frames if `render.py` changes later (version/hash in metadata)?

Check: a short findings file under `research/` with measured numbers per candidate and a recommended format (with rationale), plus the provenance metadata it must carry.

## Comments
**Resolution (2026-11-03):** Findings in [research/01-storage-and-loading.md](../research/01-storage-and-loading.md), measured with `prototypes/storage_bench.py` on 150 episodes and scaled to 3000.
- **Format:** store States, `clean`/`executed` actions, `perturbed` flags and per-episode metadata; **do not store Frames**, re-render them in the loader. 0.11 GB at 3000 episodes, no new dependency; re-rendered Frames are bit-identical to recorded ones, including in spawned workers.
- **Speed is not the constraint:** a 6.3 M-param transformer on MPS consumes about 13 windows/s; re-rendering gives 900 windows/s in one process and 3.5k/s with 4 workers, and 15k random Frames/s for Step 4.
- **Raw Frames don't fit:** 33 GB at 3000 episodes vs about 34 GB free disk and 16 GB RAM. Per-episode compressed `.npz` is slow at random access (389 windows/s); zarr v3 is slow at small reads.
- **Provenance (required):** per episode, the SHA-256 over all its Frames plus full `RenderParams`/`TrackParams`; per dataset, the `toycar` version, a hash of `render.py` + `track.py` source, and Python/numpy versions. The loader re-renders and checks a few random episodes on open; a `verify` command checks them all; mismatches raise.
- **Fallback** if stored pixels are ever needed: HDF5 + lzf, chunk = 1 episode × 17 Frames (4.2 GB, 13k windows/s), adds `h5py`.
- **Side finding for Step 6:** about 1100 tokens per window with plain attention caps the batch at 8 on this Mac (batch 16 swaps, 32 runs out of memory).
