# Storage format and loading throughput

Ticket: [01-storage-and-loading](../issues/01-storage-and-loading.md). Measured with `prototypes/storage_bench.py` (results in `prototypes/out/storage_bench_*.json`) on the project's 16 GB M-series Mac, Python 3.14, numpy 2.5, torch 2.15 (MPS), h5py 3.16, zarr 3.4. 150 expert episodes (900 ticks, default noise, 25 per Instruction × Direction), scaled linearly to 3000.

## Constraints found

- **Disk:** about 34 GB free. **RAM:** 16 GB, shared with MPS.
- Raw Frames are 12,288 B each; 3000 × 901 Frames = **33 GB**. That exceeds free disk and RAM, so any format that needs the raw array on disk or in memory is out at that size.
- One window (17 Frames) is 209 KB as uint8 or 836 KB as float32. States for 3000 episodes are 86 MB as float64, so all non-image data fits in RAM trivially.

## Results (3000-episode scale)

| Format | Disk | Write | Random windows/s, 1 process | DataLoader, 4 workers | Random Frames/s, 1 process |
|---|---|---|---|---|---|
| per-episode `.npz`, compressed, Frames included | 0.56 GB | 44 ms/ep | 389 | 215 | 428 |
| one raw uint8 `.npy` memmap | **33 GB** | 10 ms/ep | 240k\* | 65k\* | 381k\* |
| HDF5, lzf, chunk = 1 ep × 17 Frames | 4.2 GB | 11 ms/ep | 4,335 | 13,109 | 12,004 |
| HDF5, gzip-4, same chunks | 1.3 GB | 49 ms/ep | 1,936 | 7,401 | 4,729 |
| zarr v3, blosc, same chunks | 0.45 GB | 34 ms/ep | 582 | 1,415 | 1,783 |
| **States only, re-render Frames on read** | **0.11 GB** | 2 ms/ep | 902 | 3,550 | 15,207 |

\* memmap numbers come from the page cache (the 1.6 GB test file fit in RAM). At 33 GB they would be disk-bound, and the file doesn't fit on the disk anyway.

- `.npz` per episode decompresses the whole 11 MB episode on every random access. Workers make it slower (I/O and decompression contention).
- zarr v3 has high per-call overhead for small reads.
- **Re-rendering is bit-identical:** stored Frames, Frames re-rendered in the same process and Frames re-rendered in two spawned worker processes all have the same SHA-256 over one full episode. A Frame renders in 0.06 ms.

## Training budget (what the loader must keep up with)

A stand-in pre-norm transformer encoder on MPS with PLAN-sized windows (64 latent tokens × 17 Frames + state + instruction + 16 actions = 1106 tokens):

| Model | Batch 4 | Batch 8 | Batch 16 | Batch 32 |
|---|---|---|---|---|
| d256 × 8 layers (6.3 M params) | 314 ms/step, 13 windows/s | 615 ms/step, 13 windows/s | 163 s/step (swapping) | out of memory |

The d384 run was stopped: it pushed the machine into about 20 GB of swap.

- **Training consumes about 13 windows/s.** Every candidate except the raw memmap is at least 15× faster than needed, even single-process. Load speed doesn't decide the format.
- Side finding for Step 6 (not this ticket): with plain attention, about 1100 tokens per window caps the batch at 8 on this Mac. Step 6 needs fewer tokens per Frame, memory-efficient attention, or shorter futures. Logged here so Step 6's map picks it up.

## Recommendation

**Store States, actions, flags and metadata; do not store Frames. Re-render Frames in the loader.**

- Smallest by far (0.11 GB for 3000 episodes; uncompressed `.npy` per split, or one `.npz` per episode, are both fine at this size). No new dependency.
- Exact: the renderer is pure and verified bit-identical across processes.
- It makes the Step 10 visual probes free: the same episodes can be re-rendered under held-out `RenderParams` without re-recording. Per-episode visual randomisation is just a `RenderParams` value in the episode metadata.
- Fast enough: 900 windows/s in-process, 3.5k/s with 4 workers, versus about 13/s consumed. Step 4 (random Frames) gets 15k Frames/s in one process.

**Risk:** Frames now depend on the render code at load time. A later change to `render.py`, numpy or the `RenderParams` defaults would silently change the training images. Mitigation, required:

- Per episode, store the SHA-256 over all its Frames (as Step 1 recordings already store per-Frame hashes), plus the full `RenderParams` and `TrackParams` (never defaults).
- Per dataset (manifest), store the `toycar` version, a hash of `render.py` and `track.py` source, and the Python and numpy versions.
- The loader re-renders and checks the hash of a few random episodes on open (cheap). A `verify` command checks all of them. On mismatch it raises, rather than training on different images.

**Fallback** if a later step needs stored pixels (e.g. Frames that aren't reproducible, or a model on another machine without `toycar`): HDF5 with lzf, chunked one episode × 17 Frames. It's the fastest compressed option (4.2 GB at 3000 episodes, 13k windows/s with workers), at the cost of adding `h5py`. Not needed now.

**Rejected:**
- Raw memmap: 33 GB doesn't fit the disk or RAM.
- Compressed `.npz` per episode with Frames: slow random access, and the stored Frames are redundant.
- zarr: smallest compressed option, but slow small reads.

**Step 4+ latents** (out of scope): 8×8×16 latents in float16 are 2 KB per Frame, 5.5 GB at 3000 episodes. If Step 6 caches latents, that is its own storage decision.
