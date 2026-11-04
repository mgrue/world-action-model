# Throughput and size budget on MPS

Labels: wayfinder:research, wayfinder:prototype
Status: closed
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What model size, token count and attention implementation can train on this Mac in a sensible time?

- Re-measure the Step 3 stand-in with `torch.nn.functional.scaled_dot_product_attention` (memory-efficient / flash paths on MPS in torch 2.15?) at 1106 tokens: ms/step and peak memory at batch 8/16/32, d256/d384, 6–12 layers.
- The same with fewer future Frames (e.g. 4 or 8 of 16 via `future_frames`, ≈ 330–600 tokens) and with pooled Frame tokens (e.g. 2×2 → 16 tokens per Frame).
- float32 vs bfloat16 autocast on MPS for a transformer (Step 4 found no gain for convs).
- Include the on-the-fly `encode_windows` cost (≈ 6 ms per Window) and data loading.
- Translate into a budget: Windows/s, and how many hours for e.g. 1, 3, 10 passes over 100k / 797k Windows.

Check: a table of (params, tokens, batch, attention, precision) → ms/step, Windows/s, peak memory; a recommended envelope (params, tokens per Window, batch, hours) for ticket 03 and 05 to design within.

## Comments
- Measured with `prototypes/throughput_bench.py` (FLUX-like stand-in block, every config in its own subprocess, OOM instead of swap); full tables in [`research/01-throughput-budget.md`](../research/01-throughput-budget.md).
- SDPA = plain attention on MPS (no memory-efficient kernel; 1106 tokens: 515 vs 518 ms at batch 8, same 4.2 GB) → token count is the only lever. bf16 autocast 20–30% *slower* than fp32. Speed ∝ params × tokens (compute-bound); Windows/s flat above batch 16.
- Input cost adds serially (encode shares the GPU): 8.3 ms per Window with 16 future Frames, 4.4 ms with 8, 2.4 ms with 4.
- End to end at d256×8 (8.5 M): PLAN layout 1106 tokens 14.7 W/s (19 h per 1 M Windows); 2×2 patches + 16 future (290 tokens) 47 W/s (5.9 h); 2×2 + 8 future (162 tokens) 88 W/s (3.2 h). d384×8 (19 M) about 1.4–1.8× slower.

**Resolution (2026-11-04):** envelope for tickets 02, 03 and 05:
- **About 160–300 tokens per Window**, via a lossless **2×2 patchify** (16 tokens per Frame, 64 values each, FLUX-style). The 1106-token PLAN layout is out (15–19 h runs, batch ≤ 16).
- **16 or 8 future Frames** (290 or 162 tokens); the choice is ticket 02's.
- **8–20 M params** (d256×8 … d384×8). **Batch 32**, float32, SDPA, eager.
- **Full run 3–8 h** (about 1 M Windows seen, overnight with `caffeinate`); quick experiments about 100k Windows in 20–40 min.
