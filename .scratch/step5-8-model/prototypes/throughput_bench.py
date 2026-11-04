"""PROTOTYPE — throwaway. Steps 5–8 ticket 01 "Throughput and size budget on MPS".

Times one training step (forward + backward + AdamW) of a FLUX-like stand-in transformer on MPS
for a grid of (width, depth, tokens per Window, batch, attention, precision), each in its own
subprocess (an OOM or swap in one config can't poison the next). Also times the per-Window input
cost: Step 3 Window rendering and `vae-v1` encoding.

    uv run python .scratch/step5-8-model/prototypes/throughput_bench.py            # full grid
    uv run python .scratch/step5-8-model/prototypes/throughput_bench.py --inputs   # input cost only

Writes prototypes/out/bench_*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent / "out"


# -- stand-in model (FLUX single-stream block, shrunk) --------------------------------------
def build(d: int, layers: int, attn: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    heads = d // 64

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.mlp_in = nn.Linear(d, 2 * 3 * d, bias=False)  # SwiGLU, ratio 3
            self.attn_out = nn.Linear(d, d, bias=False)
            self.mlp_out = nn.Linear(3 * d, d, bias=False)
            self.qn = nn.RMSNorm(64)
            self.kn = nn.RMSNorm(64)
            self.mod = nn.Linear(d, 3 * d)

        def forward(self, x, c):  # c: (B, N, d) per-token conditioning
            shift, scale, gate = self.mod(F.silu(c)).chunk(3, -1)
            h = (1 + scale) * self.norm(x) + shift
            B, N, _ = h.shape
            q, k, v = self.qkv(h).view(B, N, 3, heads, 64).permute(2, 0, 3, 1, 4)
            q, k = self.qn(q), self.kn(k)
            if attn == "sdpa":
                a = F.scaled_dot_product_attention(q, k, v)
            else:  # plain: materialise the (N, N) attention matrix
                w = (q @ k.transpose(-1, -2)) / 8.0
                a = w.softmax(-1) @ v
            a = a.transpose(1, 2).reshape(B, N, d)
            g, u = self.mlp_in(h).chunk(2, -1)
            return x + gate * (self.attn_out(a) + self.mlp_out(F.silu(g) * u))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(16, d)
            self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.out = nn.Linear(d, 16)

        def forward(self, x, t):
            c = self.temb(t)
            h = self.inp(x)
            for b in self.blocks:
                h = b(h, c)
            return self.out(h)

    return Net()


def tokens(future: int, per_frame: int) -> int:
    """obs Frame + `future` Frames, State + Instruction tokens, 16 action tokens."""
    return per_frame * (future + 1) + 2 + 16


def one(cfg: dict) -> dict:
    import torch

    torch.manual_seed(0)
    dev = torch.device("mps")
    model = build(cfg["d"], cfg["layers"], cfg["attn"]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, N = cfg["batch"], cfg["tokens"]
    x = torch.randn(B, N, 16, device=dev)
    t = torch.randn(B, N, 256, device=dev)
    y = torch.randn(B, N, 16, device=dev)
    ac = torch.autocast("mps", dtype=torch.bfloat16) if cfg["prec"] == "bf16" else torch.autocast("mps", enabled=False)

    def step():
        with ac:
            loss = (model(x, t).float() - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.mps.synchronize()
    n = cfg.get("steps", 8)
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / n
    return {**cfg, "params": params, "ms_step": round(dt * 1e3, 1), "windows_s": round(B / dt, 1),
            "mem_gb": round(torch.mps.driver_allocated_memory() / 2**30, 2)}


def run_sub(cfg: dict, timeout: float = 180.0) -> dict:
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8"}  # error, not swap
    try:
        p = subprocess.run([sys.executable, __file__, "--one", json.dumps(cfg)], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {**cfg, "error": f"timeout > {timeout:.0f} s (swapping?)"}
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        return {**cfg, "error": next((l for l in reversed(err) if "out of memory" in l.lower()), err[-1] if err else "?")[:160]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def grid() -> list[dict]:
    cfgs = []
    full, f8, f4 = tokens(16, 64), tokens(8, 64), tokens(4, 64)
    pooled16, pooled8 = tokens(16, 16), tokens(8, 16)
    # 1. plain vs SDPA at full PLAN size (Step 3 reproduced)
    for attn in ("plain", "sdpa"):
        for b in (8, 16, 32):
            cfgs.append(dict(group="attn", d=256, layers=8, tokens=full, batch=b, attn=attn, prec="fp32"))
    # 2. token count × width/depth, SDPA fp32
    for d, L in ((256, 8), (384, 8), (384, 12), (512, 8)):
        for n in (full, f8, f4, pooled16, pooled8):
            for b in (8, 16, 32, 64):
                if n == full and b == 64:
                    continue
                cfgs.append(dict(group="size", d=d, layers=L, tokens=n, batch=b, attn="sdpa", prec="fp32"))
    # 3. bf16 autocast
    for d, L, n, b in ((256, 8, full, 16), (384, 8, f8, 32), (384, 8, pooled16, 32), (384, 12, pooled16, 32)):
        cfgs.append(dict(group="prec", d=d, layers=L, tokens=n, batch=b, attn="sdpa", prec="bf16"))
    return cfgs


def inputs() -> dict:
    """Per-Window cost of Step 3 rendering + vae-v1 encoding, batch 32, 17 and 9 Frames."""
    import numpy as np
    import torch

    from toycar.dataset import open_dataset
    from wam.compressor import load

    es = open_dataset("data/step3-v1", check=False).split("train")
    comp = load("vae-v1")
    res = {}
    for ff, label in ((None, "16 future"), (tuple(range(1, 16, 2)), "8 future"), ((3, 7, 11, 15), "4 future")):
        w = es.windows(future_frames=ff)
        idx = np.random.default_rng(0).integers(len(w), size=64)
        t0 = time.perf_counter()
        items = [w[int(i)] for i in idx]
        t_load = (time.perf_counter() - t0) / len(idx)
        batch = {k: torch.from_numpy(np.stack([it[k] for it in items[:32]])) for k in ("obs_frame", "future_frames")}
        comp.encode_windows(batch)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            comp.encode_windows(batch)
        torch.mps.synchronize()
        t_enc = (time.perf_counter() - t0) / 5 / 32
        res[label] = {"load_render_ms": round(t_load * 1e3, 2), "encode_ms": round(t_enc * 1e3, 2),
                      "windows_s_ceiling": round(1 / (t_load + t_enc), 0)}
    res["train_windows"] = len(es.windows())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--inputs", action="store_true")
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(json.loads(a.one))))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if a.inputs:
        r = inputs()
        (OUT / "bench_inputs.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))
        return 0
    failed: set[tuple] = set()  # (group, d, layers, tokens, attn, prec) that OOMed at a smaller batch
    with open(OUT / "bench_steps.jsonl", "w") as f:
        for cfg in grid():
            key = (cfg["group"], cfg["d"], cfg["layers"], cfg["tokens"], cfg["attn"], cfg["prec"])
            if key in failed:
                r = {**cfg, "error": "skipped (smaller batch failed)"}
            else:
                r = run_sub(cfg)
                if "error" in r:
                    failed.add(key)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
