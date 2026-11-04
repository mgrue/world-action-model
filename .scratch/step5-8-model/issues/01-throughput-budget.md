# Throughput and size budget on MPS

Labels: wayfinder:research, wayfinder:prototype
Status: open
Assignee: mgrue
Blocked by:
Parent: ../MAP.md

## Question

What model size, token count and attention implementation can train on this Mac in a sensible time?

- Re-measure the Step 3 stand-in with `torch.nn.functional.scaled_dot_product_attention` (memory-efficient / flash paths on MPS in torch 2.15?) at 1106 tokens: ms/step and peak memory at batch 8/16/32, d256/d384, 6–12 layers.
- The same with fewer future Frames (e.g. 4 or 8 of 16 via `future_frames`, ≈ 330–600 tokens) and with pooled Frame tokens (e.g. 2×2 → 16 tokens per Frame).
- float32 vs bfloat16 autocast on MPS for a transformer (Step 4 found no gain for convs).
- Include the on-the-fly `encode_windows` cost (≈ 6 ms per Window) and data loading.
- Translate into a budget: Windows/s, and how many hours for e.g. 1, 3, 10 passes over 100k / 1.29 M Windows.

Check: a table of (params, tokens, batch, attention, precision) → ms/step, Windows/s, peak memory; a recommended envelope (params, tokens per Window, batch, hours) for ticket 03 and 05 to design within.

## Comments
