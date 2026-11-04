"""PROTOTYPE — throwaway. Steps 5–8 ticket 03 "Transformer blocks and noise conditioning".

The world action model as decided in tickets 02–04, shrunk to one file:
- 162-token layout (obs Frame 16, State, Instruction, 8 even future Frames × 16, 16 actions), 2×2 patchify
- 3-axis RoPE (t 32 / h 16 / w 16 per 64-dim head), full attention
- FLUX parallel single-stream block, per-token timestep embedding, 5 per-Stream Modulations shared
  across blocks, FLUX LastLayer per predicted Stream, adaLN-Zero init
- flow loss: one t per Window, logit-normal + shift 1.65, per-stream means, actions standardised

    uv run --with einops python .scratch/step5-8-model/prototypes/model_proto.py            # param count, ms/step, overfit run
    uv run --with einops python .scratch/step5-8-model/prototypes/model_proto.py --quick    # tiny arch on CPU, smoke only

Writes prototypes/out/model_proto.json and model_proto_loss.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

OUT = Path(__file__).parent / "out"
H, STRIDE, HEAD = 16, 2, 64
ACT_MEAN = torch.tensor([0.0, 0.077])
ACT_STD = torch.tensor([0.618, 0.373])
STREAMS = ("obs", "state", "instr", "fut", "act")


# -- positions --------------------------------------------------------------------------------
def position_ids(horizon: int = H, stride: int = STRIDE, video: bool = True) -> torch.Tensor:
    """(N, 3) integer (t, h, w) per token, in the ticket-02 order obs, State, Instruction, fut, act."""
    grid = torch.cartesian_prod(torch.arange(4), torch.arange(4))  # 16 × (h, w), row-major
    rows = [torch.cat([torch.zeros(16, 1, dtype=torch.long), grid], 1),  # obs Frame at t = 0
            torch.zeros(2, 3, dtype=torch.long)]                         # State, Instruction
    if video:
        for k in range(stride, horizon + 1, stride):                     # future Frame k at t = k
            rows.append(torch.cat([torch.full((16, 1), k), grid], 1))
    rows.append(torch.stack([torch.arange(1, horizon + 1), torch.zeros(horizon, dtype=torch.long),
                             torch.zeros(horizon, dtype=torch.long)], 1))  # action k at t = k
    return torch.cat(rows)


def rope_freqs(ids: torch.Tensor, axes=(32, 16, 16), theta: float = 10000.0) -> torch.Tensor:
    """(N, 3) → (N, HEAD/2) complex rotations, one block of frequencies per axis."""
    parts = []
    for i, dim in enumerate(axes):
        omega = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        parts.append(ids[:, i, None].double() * omega[None])
    ang = torch.cat(parts, -1).float()
    return torch.stack([ang.cos(), ang.sin()], -1)  # (N, HEAD/2, 2)


def apply_rope(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    c, s = f[..., 0], f[..., 1]
    out = torch.stack([x2[..., 0] * c - x2[..., 1] * s, x2[..., 0] * s + x2[..., 1] * c], -1)
    return out.flatten(-2).type_as(x)


# -- model --------------------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[..., None].float() * freqs
    return torch.cat([args.cos(), args.sin()], -1)


class Block(nn.Module):
    """FLUX single-stream block: parallel attention + SwiGLU, one gate, QK RMSNorm, no biases."""

    def __init__(self, d: int, mlp_ratio: float = 3.0) -> None:
        super().__init__()
        self.heads, self.hid = d // HEAD, int(d * mlp_ratio)
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.mlp_in = nn.Linear(d, 2 * self.hid, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.mlp_out = nn.Linear(self.hid, d, bias=False)
        self.qn, self.kn = nn.RMSNorm(HEAD, eps=1e-6), nn.RMSNorm(HEAD, eps=1e-6)

    def forward(self, x, mod, pe):
        shift, scale, gate = mod
        h = (1 + scale) * self.norm(x) + shift
        B, N, d = h.shape
        q, k, v = rearrange(self.qkv(h), "b n (three h e) -> three b h n e", three=3, h=self.heads)
        q, k = apply_rope(self.qn(q), pe), apply_rope(self.kn(k), pe)
        a = rearrange(F.scaled_dot_product_attention(q, k, v), "b h n e -> b n (h e)")
        g, u = self.mlp_in(h).chunk(2, -1)
        return x + gate * (self.attn_out(a) + self.mlp_out(F.silu(g) * u))


class LastLayer(nn.Module):
    def __init__(self, d: int, out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(d, 2 * d, bias=False)
        self.lin = nn.Linear(d, out, bias=False)

    def forward(self, x, vec):
        shift, scale = self.ada(F.silu(vec)).chunk(2, -1)
        return self.lin((1 + scale) * self.norm(x) + shift)


class WAM(nn.Module):
    def __init__(self, d: int = 256, depth: int = 8, horizon: int = H, stride: int = STRIDE, video: bool = True) -> None:
        super().__init__()
        self.d, self.horizon, self.stride, self.video = d, horizon, stride, video
        self.frame_in = nn.Linear(64, d)            # shared obs / future Frame tokens
        self.state_in = nn.Linear(5, d)
        self.instr = nn.Embedding(4, d)             # outer, inner, any, null
        self.act_in = nn.Linear(2, d)
        self.t_in = nn.Sequential(nn.Linear(256, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))
        self.mods = nn.ModuleDict({s: nn.Linear(d, 3 * d, bias=False) for s in STREAMS})
        self.blocks = nn.ModuleList(Block(d) for _ in range(depth))
        self.fut_out = LastLayer(d, 64) if video else None
        self.act_out = LastLayer(d, 2)
        self.register_buffer("pe", rope_freqs(position_ids(horizon, stride, video)), persistent=False)
        self.n_fut = 16 * (horizon // stride) if video else 0
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.instr.weight, std=0.02)
        for m in self.t_in:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for lin in self.mods.values():                          # adaLN-Zero: blocks start as identity
            nn.init.zeros_(lin.weight)
        for last in (self.fut_out, self.act_out):
            if last is not None:
                nn.init.zeros_(last.ada.weight)
                nn.init.zeros_(last.lin.weight)                 # initial velocity = 0

    def forward(self, obs, state, instr, x_fut, x_act, t):
        """obs (B,16,8,8) · state (B,5) · instr (B,) · x_fut (B,K,16,8,8) or None · x_act (B,H,2) · t (B,)."""
        B = obs.shape[0]
        patch = lambda z: rearrange(z, "... c (h ph) (w pw) -> ... (h w) (c ph pw)", ph=2, pw=2)
        toks = [self.frame_in(patch(obs)), self.state_in(state)[:, None], self.instr(instr)[:, None]]
        lens = [16, 1, 1]
        if self.video:
            toks.append(self.frame_in(patch(x_fut).flatten(1, 2)))
            lens.append(self.n_fut)
        toks.append(self.act_in(x_act))
        lens.append(self.horizon)
        x = torch.cat(toks, 1)
        streams = ["obs", "state", "instr"] + (["fut"] if self.video else []) + ["act"]
        # per-token noise level → vec; clean Streams at t = 0
        t_tok = torch.cat([torch.zeros(B, 18, device=x.device)] +
                          [t[:, None].expand(B, n) for n in lens[3:]], 1)
        vec = self.t_in(timestep_embedding(t_tok))                       # (B, N, d)
        vec_s = vec.split(lens, 1)
        mod = torch.cat([self.mods[s](F.silu(v)) for s, v in zip(streams, vec_s)], 1).chunk(3, -1)
        for b in self.blocks:
            x = b(x, mod, self.pe)
        xs, vs = x.split(lens, 1), vec_s
        v_act = self.act_out(xs[-1], vs[-1])
        v_fut = None
        if self.video:
            K = self.n_fut // 16
            v_fut = rearrange(self.fut_out(xs[3], vs[3]), "b (k h w) (c ph pw) -> b k c (h ph) (w pw)",
                              k=K, h=4, w=4, ph=2, pw=2)
        return v_fut, v_act


def sample_t(B: int, shift: float = 1.65, gen=None) -> torch.Tensor:
    s = torch.sigmoid(torch.randn(B, generator=gen))
    return shift * s / (1 + (shift - 1) * s)


def flow_loss(model: WAM, batch: dict, gen=None, p_uncond: float = 0.1):
    obs, fut, state, instr, act = (batch[k] for k in ("obs", "fut", "state", "instr", "act"))
    B, dev = obs.shape[0], obs.device
    a_n = (act - ACT_MEAN.to(dev)) / ACT_STD.to(dev)
    t = sample_t(B, gen=gen).to(dev)
    instr = torch.where(torch.rand(B, generator=gen).to(dev) < p_uncond, torch.full_like(instr, 3), instr)
    ea = torch.randn(a_n.shape, generator=gen).to(dev)
    x_a = t[:, None, None] * ea + (1 - t[:, None, None]) * a_n
    x_f, ef = None, None
    if model.video:
        ef = torch.randn(fut.shape, generator=gen).to(dev)
        tt = t.view(B, 1, 1, 1, 1)
        x_f = tt * ef + (1 - tt) * fut
    v_f, v_a = model(obs, state, instr, x_f, x_a, t)
    la = ((v_a - (ea - a_n)) ** 2).mean()
    lv = ((v_f - (ef - fut)) ** 2).mean() if model.video else torch.zeros((), device=dev)
    hi = t > 0.7
    la_tok = ((v_a - (ea - a_n)) ** 2).mean((1, 2)).detach()
    return lv + la, {"loss_video": lv.item(), "loss_action": la.item(),
                     "loss_action_hi_t": la_tok[hi].mean().item() if hi.any() else float("nan"),
                     "loss_action_lo_t": la_tok[~hi].mean().item() if (~hi).any() else float("nan")}


# -- data ---------------------------------------------------------------------------------------
def load_windows(n: int, device, seed: int = 0) -> dict:
    from toycar.dataset import open_dataset
    from wam.compressor import load

    ds = open_dataset("data/step3-v1", check=False)
    w = ds.split("train").windows(future_frames=tuple(range(STRIDE - 1, H, STRIDE)))
    idx = np.sort(np.random.default_rng(seed).choice(len(w), n, replace=False))
    items = [w[int(i)] for i in idx]
    stack = lambda k: np.stack([it[k] for it in items])
    comp = load("vae-v1", device=device)
    obs, fut = comp.encode_windows({"obs_frame": stack("obs_frame"), "future_frames": stack("future_frames")})
    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return {"obs": obs.clone(), "fut": fut.clone(), "state": to(stack("state")),
            "instr": to(stack("instruction"), torch.long), "act": to(stack("actions"))}


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cpu" if args.quick or not torch.backends.mps.is_available() else "mps")
    d, depth = (64, 2) if args.quick else (256, 8)
    res: dict = {"device": str(dev), "d": d, "depth": depth}

    # parameter count, by part, and the variant layouts
    m = WAM(d, depth).to(dev)
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    res["params"] = {"total": n(m), "blocks": n(m.blocks), "modulations": n(m.mods), "t_in": n(m.t_in),
                     "in_projections": n(m.frame_in) + n(m.state_in) + n(m.instr) + n(m.act_in),
                     "last_layers": n(m.fut_out) + n(m.act_out)}
    res["tokens"] = {"default": len(position_ids()), "stride1": len(position_ids(stride=1)),
                     "no_video": len(position_ids(video=False)), "h8": len(position_ids(8)), "h4": len(position_ids(4))}
    print(json.dumps(res, indent=1))

    # smoke: every switch combination runs forward + backward
    B = 4
    obs, st, ins = torch.randn(B, 16, 8, 8, device=dev), torch.randn(B, 5, device=dev), torch.randint(0, 4, (B,), device=dev)
    for hz, vid in ((16, True), (8, True), (4, True), (16, False)):
        mm = WAM(d, depth, horizon=hz, video=vid).to(dev)
        xf = torch.randn(B, hz // 2, 16, 8, 8, device=dev) if vid else None
        vf, va = mm(obs, st, ins, xf, torch.randn(B, hz, 2, device=dev), torch.rand(B, device=dev))
        assert va.shape == (B, hz, 2) and (vf is None or vf.shape == xf.shape)
        assert va.abs().max() == 0, "zero-init head must output 0"
        (va.sum() + (vf.sum() if vf is not None else 0)).backward()
    # regression path: zero action inputs at t = 1
    mm = WAM(d, depth, video=False).to(dev)
    _, va = mm(obs, st, ins, None, torch.zeros(B, 16, 2, device=dev), torch.ones(B, device=dev))
    res["smoke"] = "ok"

    # ms/step at batch 32 (random inputs, model only; vae encode adds ~4.4 ms/Window, ticket 01)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    Bt = 32
    fake = {"obs": torch.randn(Bt, 16, 8, 8, device=dev), "fut": torch.randn(Bt, 8, 16, 8, 8, device=dev),
            "state": torch.randn(Bt, 5, device=dev), "instr": torch.randint(0, 3, (Bt,), device=dev),
            "act": torch.rand(Bt, 16, 2, device=dev) * 2 - 1}
    for i in range(13):
        if i == 3:
            sync(dev); t0 = time.perf_counter()
        loss, _ = flow_loss(m, fake)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(dev)
    ms = (time.perf_counter() - t0) / 10 * 1000
    res["ms_per_step_b32"] = round(ms, 1)
    res["model_windows_per_s"] = round(Bt / ms * 1000, 1)
    res["end_to_end_windows_per_s_est"] = round(1000 / (ms / Bt + 4.4), 1)
    print("ms/step b32:", res["ms_per_step_b32"], "model W/s:", res["model_windows_per_s"],
          "end-to-end est W/s:", res["end_to_end_windows_per_s_est"])

    # overfit a few hundred real train Windows: both losses must fall
    if args.quick:
        args.windows, args.steps = 64, 60
    t0 = time.perf_counter()
    data = load_windows(args.windows, dev)
    res["load_encode_s"] = round(time.perf_counter() - t0, 1)
    torch.manual_seed(0)
    m = WAM(d, depth).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    rows, ema = [], None
    for step in range(args.steps):
        idx = torch.randint(0, args.windows, (Bt,), generator=gen).to(dev)
        batch = {k: v[idx] for k, v in data.items()}
        loss, logs = flow_loss(m, batch, gen=gen)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        opt.step()
        rows.append({"step": step, **logs, "grad_norm": gn})
        if step % 100 == 0 or step == args.steps - 1:
            print(step, {k: round(v, 4) for k, v in logs.items()})

    def window_mean(key, lo, hi):
        v = [r[key] for r in rows[lo:hi] if not math.isnan(r[key])]
        return round(float(np.mean(v)), 4)
    k = min(100, args.steps // 5)
    res["overfit"] = {"windows": args.windows, "steps": args.steps, "batch": Bt, "lr": 3e-4,
                      **{f"{key}_first{k}": window_mean(key, 0, k) for key in ("loss_video", "loss_action", "loss_action_hi_t", "loss_action_lo_t")},
                      **{f"{key}_last{k}": window_mean(key, -k, None) for key in ("loss_video", "loss_action", "loss_action_hi_t", "loss_action_lo_t")}}
    OUT.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    (OUT / f"model_proto{tag}.json").write_text(json.dumps(res, indent=1))
    with open(OUT / f"model_proto{tag}_loss.csv", "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(json.dumps(res["overfit"], indent=1))


if __name__ == "__main__":
    main()
