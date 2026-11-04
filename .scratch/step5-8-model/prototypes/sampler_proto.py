"""PROTOTYPE — throwaway. Steps 5–8 ticket 06 "Sampler": Euler + per-Stream CFG on the ticket-03 prototype
model, timed on MPS (cost is independent of the weights, so an untrained model is enough).

    uv run --with einops python .scratch/step5-8-model/prototypes/sampler_proto.py
Writes prototypes/out/sampler_proto.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from model_proto import WAM, ACT_MEAN, ACT_STD, sync  # noqa: E402

def timeshift(a, t):
    return a * t / (1 + (a - 1) * t)

@torch.no_grad()
def sample(model, obs, state, instr, *, n_steps=10, cfg_action=1.0, cfg_video=1.0, t_shift=1.65, gen=None):
    B, dev = obs.shape[0], obs.device
    K = model.n_fut // 16
    x_a = torch.randn(B, model.horizon, 2, generator=gen).to(dev)
    x_z = torch.randn(B, K, 16, 8, 8, generator=gen).to(dev) if model.video else None
    cfg = cfg_action != 1.0 or cfg_video != 1.0
    ts = timeshift(t_shift, torch.linspace(1, 0, n_steps + 1)).tolist()
    for t, tn in zip(ts[:-1], ts[1:]):
        tt = torch.full((B,), t, device=dev)
        if cfg:   # one batched forward of 2B: conditional then null Instruction
            cat = lambda x: None if x is None else torch.cat([x, x])
            vz, va = model(cat(obs), cat(state), torch.cat([instr, torch.full_like(instr, 3)]), cat(x_z), cat(x_a), cat(tt))
            va = va[B:] + cfg_action * (va[:B] - va[B:])
            vz = None if vz is None else vz[B:] + cfg_video * (vz[:B] - vz[B:])
        else:
            vz, va = model(obs, state, instr, x_z, x_a, tt)
        x_a = x_a + (tn - t) * va
        if model.video:
            x_z = x_z + (tn - t) * vz
    acts = (x_a * ACT_STD.to(dev) + ACT_MEAN.to(dev)).clamp(-1, 1)
    return x_z, acts

def timed(fn, reps):
    fn(); fn(); sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / reps * 1000

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
res = {"device": str(dev)}
for video in (True, False):
    m = WAM(256, 8, video=video).to(dev).eval()
    for p in m.parameters():                     # non-zero weights so nothing short-circuits
        torch.nn.init.normal_(p, std=0.02)
    for B, steps, cfg in ((1, 10, 1.0), (1, 10, 2.0), (1, 4, 1.0), (1, 20, 1.0), (16, 10, 1.0), (256, 10, 1.0)):
        obs, st = torch.randn(B, 16, 8, 8, device=dev), torch.randn(B, 5, device=dev)
        ins = torch.zeros(B, dtype=torch.long, device=dev)
        g = torch.Generator().manual_seed(0)
        ms = timed(lambda: sample(m, obs, st, ins, n_steps=steps, cfg_action=cfg, cfg_video=cfg, gen=g), 5 if B < 256 else 2)
        key = f"{'video' if video else 'novideo'}_B{B}_steps{steps}_cfg{cfg}"
        res[key] = {"ms_per_call": round(ms, 1), "ms_per_chunk": round(ms / B, 2)}
        print(key, res[key])
# determinism: same seed → same chunk (same device)
m = WAM(256, 8).to(dev).eval()
for p in m.parameters():
    torch.nn.init.normal_(p, std=0.02)
obs, st, ins = torch.randn(1, 16, 8, 8, device=dev), torch.randn(1, 5, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
a1 = sample(m, obs, st, ins, gen=torch.Generator().manual_seed(7))[1]
a2 = sample(m, obs, st, ins, gen=torch.Generator().manual_seed(7))[1]
res["same_seed_identical"] = bool(torch.equal(a1, a2))
# vae-v1 encode of one Frame (Step 9 per replan)
from wam.compressor import load
import numpy as np
comp = load("vae-v1", device=dev)
fr = np.zeros((1, 64, 64, 3), np.uint8)
res["vae_encode_1_frame_ms"] = round(timed(lambda: comp.encode(fr), 20), 2)
print("same seed identical:", res["same_seed_identical"], "encode 1 frame ms:", res["vae_encode_1_frame_ms"])
Path(__file__).parent.joinpath("out/sampler_proto.json").write_text(json.dumps(res, indent=1))
