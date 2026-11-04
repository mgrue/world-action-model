# Throughput and size budget on MPS (ticket 01)

Measured 2026-11-04 on the Apple M1 Pro (16 GB, MPS ceiling 11.8 GB), torch 2.15.0, Python 3.14.
Script: `prototypes/throughput_bench.py`; raw results: `prototypes/out/bench_steps.jsonl`, `bench_inputs.json`.

## Method

- **Stand-in model:** a shrunk FLUX single-stream block: per-token shift/scale/gate modulation, parallel attention + SwiGLU MLP (ratio 3), QK RMSNorm, head size 64. Input projection 16 → d, per-token timestep MLP, output head d → 16.
- **Timing:** one full training step (forward, backward, AdamW) on random tensors. 3 warm-up steps, then 8 timed, with `torch.mps.synchronize()`.
- **Isolation:** each config runs in its own subprocess, with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` so it hits OOM instead of swapping. Timeout 180 s.
- **Token counts per Window** (obs Frame + K future Frames, plus State, Instruction and 16 action tokens):

  | Layout | Tokens |
  |---|---|
  | 64 tokens per Frame, 16 future | 1106 |
  | 64 per Frame, 8 future | 594 |
  | 64 per Frame, 4 future | 338 |
  | 2×2 patches (16 per Frame), 16 future | 290 |
  | 2×2 patches, 8 future | 162 |

- **Input cost:** Step 3 Window rendering plus `vae-v1.encode_windows`, real data, batch 32.

## Results

### Attention implementation: SDPA gives nothing on MPS

At 1106 tokens (d256 × 8, 8.5 M params) plain attention and `scaled_dot_product_attention` are identical: 515 vs 518 ms/step at batch 8, 4.2 GB either way. Batch 32 runs out of memory with both. MPS has no flash or memory-efficient kernel here. **Attention memory stays O(N²), so the number of tokens is the lever.**

### Precision: bfloat16 autocast is slower

| Config | fp32 | bf16 |
|---|---|---|
| d256×8, 1106 tok, b16 | 16.8 W/s | 12.6 W/s |
| d384×8, 594 tok, b32 | 20.0 W/s | 15.2 W/s |
| d384×8, 290 tok, b32 | 47.4 W/s | 33.3 W/s |
| d384×12, 290 tok, b32 | 30.8 W/s | 22.5 W/s |

bf16 is 20–30% slower and saves less memory than expected. Stay in float32, as Step 4 did.

### Model throughput (fp32, SDPA), Windows/s at the best batch

| Model | 1106 tok | 594 | 338 | 290 | 162 |
|---|---|---|---|---|---|
| d256×8, 8.5 M | 16.8 (b16, 8.2 GB) | 36.8 (b32) | 67.0 (b16) | 79.5 (b64) | 146 (b64) |
| d384×8, 19.1 M | 9.3 (b8; b16 swaps) | 20.0 (b32, 10 GB) | 38.1 (b32) | 47.4 (b32) | 88.6 (b64) |
| d384×12, 28.6 M | 6.5 (b8) | 13.9 (b16) | 26.3 (b32) | 30.8 (b32) | 59.6 (b64) |
| d512×8, 34.0 M | 6.6 (b8) | 14.0 (b16) | 26.9 (b32) | 31.4 (b32) | 56.5 (b64) |

- The cost is roughly proportional to params × tokens: compute-bound, about 1 TFLOP/s effective.
- Batch hardly changes Windows/s beyond 16. It sets memory and gradient noise, not speed.
- Memory limits (OOM above 11.8 GB): 1106 tokens allow batch 16 at most (d256) or 8 (≥ d384). At 290 tokens, batch 32–64 fits up to about 20 M params.

### Input cost (render + encode), per Window

| Future Frames | Load + render | `vae-v1` encode | Ceiling alone |
|---|---|---|---|
| 16 (17 Frames encoded) | 2.3 ms | 6.0 ms | 120 W/s |
| 8 (9 Frames) | 1.2 ms | 3.2 ms | 230 W/s |
| 4 (5 Frames) | 0.6 ms | 1.8 ms | 411 W/s |

The VAE encode shares the GPU with the model, so its cost **adds** to the step time; it isn't hidden. Rendering could move into DataLoader workers, but it is the smaller part. Patchifying doesn't reduce the encode cost: every Frame used must still be encoded.

### End to end (model + input, serial)

| Candidate | Params | ms per Window | Windows/s | Hours for 1 M Windows seen |
|---|---|---|---|---|
| A: 64 tok/Frame, 16 future (PLAN as written), d256×8 | 8.5 M | 59.7 + 8.3 = 68.0 | 14.7 | 18.9 |
| B: 64 tok/Frame, 8 future, d256×8 | 8.5 M | 27.9 + 4.4 = 32.2 | 31.0 | 9.0 |
| C: 64 tok/Frame, 4 future, d256×8 | 8.5 M | 14.9 + 2.4 = 17.4 | 57.6 | 4.8 |
| **D: 2×2 patches, 16 future, d256×8** | 8.5 M | 12.9 + 8.3 = 21.2 | 47.2 | **5.9** |
| **E: 2×2 patches, 8 future, d256×8** | 8.5 M | 7.0 + 4.4 = 11.4 | 88.0 | **3.2** |
| F: 2×2 patches, 16 future, d384×8 | 19.1 M | 21.1 + 8.3 = 29.4 | 34.0 | 8.2 |
| G: 2×2 patches, 8 future, d384×8 | 19.1 M | 11.6 + 4.4 = 15.9 | 62.9 | 4.4 |
| H: 2×2 patches, 16 future, d384×12 | 28.6 M | 32.4 + 8.3 = 40.7 | 24.5 | 11.3 |

For scale: train holds 797,262 Windows, but consecutive Windows overlap in 15 of 16 ticks, so the useful variety is far smaller. "1 M Windows seen" means, for example, 31k steps × batch 32.

## Recommended envelope (for tickets 02, 03 and 05)

- **Tokens per Window: about 160–300.** 2×2 patchify the Latent grid: 16 tokens per Frame, each covering 10 × 10 m, with 4 × 16 = 64 input values per token. This is lossless and is what FLUX does. It cuts the attention sequence 4× at no information cost. PLAN's 1106-token layout is affordable only as a 15–19 h run.
- **Future Frames:** all 16 (290 tokens) is affordable, about 6 h per 1 M Windows. Every second one (8, 162 tokens) halves that, and also halves the encode cost. This decision belongs to ticket 02: does every action k need its own Frame k?
- **Model size: 8–20 M params**, d256×8 to d384×8 (PLAN asks for 5–20 M). Deeper or wider than d384×8 costs proportionally more time for unknown gain.
- **Batch 32** (fits everywhere in the envelope, about 3–5 GB). Speed is flat in batch size.
- **float32, SDPA, eager.** No bf16 (slower) and no compile (Step 4 found it slower).
- **Budget:** a full training run of about **3–8 h** (overnight-sized, with `caffeinate`), seeing roughly 1 M Windows. Quick experiments at about 100k Windows take 20–40 min. Step 10's variants (baseline, no video loss, CFG) multiply this, so prefer the cheaper end (E or D at d256).
