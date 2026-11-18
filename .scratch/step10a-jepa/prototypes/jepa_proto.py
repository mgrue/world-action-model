"""Ticket 03 prototype: a small I-JEPA on 64 × 64 Frames, scored with the Pose readout and collapse diagnostics.

    uv run python .scratch/step10a-jepa/prototypes/jepa_proto.py --name base [--steps 30000] [--car partial|off|full]
        [--width 192 --depth 6 --heads 3] [--pred-width 96 --pred-depth 4] [--out-dim 32] [--vicreg 0.0]
        [--batch 128] [--lr 5e-4] [--eval-every 1000] [--readout]

Throwaway. Starting recipe from research/02 ("Recommended starting recipe"). Writes
prototypes/out/jepa_<name>/{log.csv, diag.jsonl, result.json, target.pt}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toycar.dataset import open_dataset
from wam import carcheck as cc
from wam import compressor as C

sys.path.insert(0, str(Path(__file__).parent))
import probe_vae as PV  # noqa: E402  (sample_frames, render, labels, Ridge, errors, stats, mlp_probe, LAMBDAS)

OUT = Path(__file__).parent / "out"
G = 8  # grid side (8 × 8 patches of 8 px)
TARGET_SHAPES = [(3, 3), (3, 4), (4, 3)]  # scale 0.14–0.19, aspect 0.75–1.33


# -- model ---------------------------------------------------------------------------------------
def rope_tables(head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for each of the 64 grid positions: half the head dim rotates with the row, half with the column."""
    q = head_dim // 4  # pairs per axis
    freqs = 1.0 / (100.0 ** (torch.arange(q, dtype=torch.float32) / q))
    r = torch.arange(G, dtype=torch.float32)
    rr, cc_ = torch.meshgrid(r, r, indexing="ij")
    ang = torch.cat([rr.reshape(-1, 1) * freqs, cc_.reshape(-1, 1) * freqs], 1)  # (64, head_dim/2)
    return ang.cos().to(device), ang.sin().to(device)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x (B, H, N, D); cos/sin (B, 1, N, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, w: int, heads: int) -> None:
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.proj = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))

    def forward(self, x, cos, sin):
        B, N, W = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, W // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, W)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, width: int, depth: int, heads: int, out_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.embed = nn.Conv2d(3, width, 8, 8)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)
        self.register_buffer("cos", torch.zeros(0), persistent=False)

    def tables(self, device):
        if self.cos.numel() == 0 or self.cos.device != device:
            self.cos, self.sin = rope_tables(self.blocks[0].qkv.in_features // self.heads, device)
        return self.cos, self.sin

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """x float (B, 3, 64, 64) in [-1, 1]; keep (B, K) token indices or None (all 64). Returns (B, K, out_dim)."""
        t = self.embed(x).flatten(2).transpose(1, 2)  # (B, 64, W)
        cos, sin = self.tables(x.device)
        if keep is None:
            keep = torch.arange(G * G, device=x.device).expand(len(x), -1)
        t = torch.gather(t, 1, keep[..., None].expand(-1, -1, t.shape[-1]))
        c, s = cos[keep][:, None], sin[keep][:, None]
        for b in self.blocks:
            t = b(t, c, s)
        return self.head(self.norm(t))


class Predictor(nn.Module):
    def __init__(self, dim: int, width: int, depth: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.inp, self.out = nn.Linear(dim, width), nn.Linear(width, dim)
        self.mask = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.register_buffer("cos", torch.zeros(0), persistent=False)

    def forward(self, ctx: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        """ctx (B, K, dim) at ctx_idx (B, K); tgt_idx (B, T). Returns predictions (B, T, dim)."""
        if self.cos.numel() == 0 or self.cos.device != ctx.device:
            self.cos, self.sin = rope_tables(self.inp.out_features // self.heads, ctx.device)
        B, T = tgt_idx.shape
        x = torch.cat([self.inp(ctx), self.mask.expand(B, T, -1)], 1)
        idx = torch.cat([ctx_idx, tgt_idx], 1)
        c, s = self.cos[idx][:, None], self.sin[idx][:, None]
        for b in self.blocks:
            x = b(x, c, s)
        return self.out(self.norm(x[:, -T:]))


def target_features(enc: Encoder, x: torch.Tensor) -> torch.Tensor:
    f = enc(x)
    return F.layer_norm(f, (f.shape[-1],))


# -- masking -------------------------------------------------------------------------------------
def car_patches(car_px: np.ndarray) -> np.ndarray:
    """(B, 64, 64) bool car-region pixels → (B, 64) bool patches touched."""
    return car_px.reshape(-1, G, 8, G, 8).any((2, 4)).reshape(-1, G * G)


def block_mask(r: int, c: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((G, G), bool)
    m[r : r + h, c : c + w] = True
    return m.reshape(-1)


def sample_masks(car: np.ndarray, rng: np.random.Generator, mode: str, p_car: float = 0.5,
                 n_targets: int = 4, shapes=TARGET_SHAPES, min_ctx: int = 8):
    """I-JEPA multi-block masks on the 8 × 8 grid, with an optional car bias.

    One target-block shape and one context-block side per batch; positions per Frame.
    mode: off (pure I-JEPA) | partial (with prob p_car one target covers ≥1 but not all car patches,
    and at least one car patch stays in the context) | full (with prob p_car, half the targets cover all car patches).
    Returns ctx_idx (B, K) and tgt_idx (B, n_targets, T) as int64 arrays, plus per-Frame flags.
    """
    B = len(car)
    h, w = shapes[rng.integers(len(shapes))]
    side = int(rng.choice([7, 8]))  # context scale 0.77 / 1.0 (square; 0.85–1.0 rounds to these)
    T = h * w
    positions = [(r, c) for r in range(G - h + 1) for c in range(G - w + 1)]
    blocks = np.stack([block_mask(r, c, h, w) for r, c in positions])  # (P, 64)
    cpos = [(r, c) for r in range(G - side + 1) for c in range(G - side + 1)]
    ctx_blocks = np.stack([block_mask(r, c, side, side) for r, c in cpos])
    tgts = np.zeros((B, n_targets, G * G), bool)
    ctxs = np.zeros((B, G * G), bool)
    biased = np.zeros(B, bool)
    for i in range(B):
        cp = car[i]
        ncar = int(cp.sum())
        use_bias = mode != "off" and ncar >= (2 if mode == "partial" else 1) and rng.random() < p_car
        biased[i] = use_bias
        for _attempt in range(50):
            cb = ctx_blocks[rng.integers(len(ctx_blocks))]
            ts = []
            if use_bias and mode == "partial":
                cover = (blocks & cp).sum(1)
                ok = np.nonzero((cover >= 1) & (cover < ncar))[0]
                ts.append(blocks[rng.choice(ok)] if len(ok) else blocks[rng.integers(len(blocks))])
            if use_bias and mode == "full":
                cover = (blocks & cp).sum(1)
                ok = np.nonzero(cover == ncar)[0]
                for _ in range(n_targets // 2):
                    ts.append(blocks[rng.choice(ok)] if len(ok) else blocks[rng.integers(len(blocks))])
            while len(ts) < n_targets:
                ts.append(blocks[rng.integers(len(blocks))])
            union = np.any(ts, 0)
            ctx = cb & ~union
            if ctx.sum() < min_ctx:
                continue
            if use_bias and mode == "partial" and not (ctx & cp).any():
                continue
            break
        tgts[i] = np.stack(ts)
        ctxs[i] = ctx
    K = int(ctxs.sum(1).min())
    ctx_idx = np.stack([rng.permutation(np.nonzero(m)[0])[:K] for m in ctxs])  # trim to the batch minimum
    ctx_idx.sort(1)
    tgt_idx = np.stack([[np.nonzero(t)[0] for t in tb] for tb in tgts])  # (B, n_targets, T)
    car_hidden = ~(ctxs & car).any(1) & car.any(1)  # before trimming
    return ctx_idx, tgt_idx, biased, car_hidden


# -- diagnostics ---------------------------------------------------------------------------------
def encode_all(enc: Encoder, frames_u8: np.ndarray, dev, bs: int = 1024) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(frames_u8), bs):
            x = C.to_x(torch.from_numpy(frames_u8[i : i + bs]).to(dev))
            out.append(target_features(enc, x).cpu())
    return torch.cat(out).numpy().astype(np.float64)  # (N, 64, D)


def eff_rank(X: np.ndarray) -> float:
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def diagnostics(f: np.ndarray, car: np.ndarray, st: np.ndarray, split: int) -> dict:
    """f (N, 64, D) target features of fixed eval Frames; car (N, 64) bool; st (N, 4) States.
    The first `split` Frames fit the quick ridge, the rest score it."""
    N, P, D = f.shape
    tok = f.reshape(-1, D)
    gstd = tok.std(0)
    mu_p = f.mean(0)  # (64, D)
    afstd = f.std(0)  # (64, D) across Frames at fixed position
    # car contrast vs the background distribution at the same position (Frames whose car is far away)
    far = np.zeros((N, P), bool)
    rc = np.stack(np.divmod(np.arange(P), G), 1)
    for i in range(N):
        cpos = rc[car[i]]
        if len(cpos) == 0:
            far[i] = True
            continue
        d = np.abs(rc[:, None, :] - cpos[None]).max(-1).min(1)
        far[i] = d >= 2
    contrast = []
    for p in range(P):
        bg = f[far[:, p], p]
        ci = np.nonzero(car[:, p])[0]
        if len(bg) < 50 or len(ci) == 0:
            continue
        m = bg.mean(0)
        scale = math.sqrt(float(np.trace(np.cov(bg.T)))) + 1e-8
        contrast += list(np.linalg.norm(f[ci, p] - m, axis=1) / scale)
    pc = (f - mu_p).reshape(-1, D)
    X = f.reshape(N, -1)
    Y = np.stack([st[:, 0], st[:, 1], np.cos(st[:, 2]), np.sin(st[:, 2])], 1)
    best = None
    for lam in (1.0, 10.0, 100.0):
        r = PV.Ridge().fit(X[:split], Y[:split], lam)
        pos, head = PV.errors(r.predict(X[split:]), st[split:])
        s = (float(np.percentile(pos, 95)), float(np.percentile(head, 95)))
        if best is None or s[0] / 0.3 + s[1] / 10 < best[0] / 0.3 + best[1] / 10:
            best = s
    return {
        "gstd_min": float(gstd.min()), "gstd_med": float(np.median(gstd)),
        "afstd_mean": float(afstd.mean()), "afstd_chan_min": float(afstd.mean(0).min()),
        "afstd_chan_med": float(np.median(afstd.mean(0))),
        "car_contrast": float(np.mean(contrast)) if contrast else float("nan"),
        "rank": eff_rank(tok[:: max(1, len(tok) // 20000)]), "rank_poscentred": eff_rank(pc[:: max(1, len(pc) // 20000)]),
        "ridge_pos_p95": best[0], "ridge_head_p95": best[1],
    }


# -- training ------------------------------------------------------------------------------------
def cosine(s: int, steps: int, lo: float, hi: float) -> float:
    return lo + (hi - lo) * 0.5 * (1 - math.cos(math.pi * min(s / steps, 1.0)))


def lr_at(s: int, steps: int, peak: float, start: float = 1e-4, final: float = 1e-6) -> float:
    wu = int(0.05 * steps)
    if s < wu:
        return start + (peak - start) * s / wu
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, steps - wu)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--car", choices=["off", "partial", "full"], default="partial")
    ap.add_argument("--p-car", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--pred-width", type=int, default=96)
    ap.add_argument("--pred-depth", type=int, default=4)
    ap.add_argument("--out-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ema", type=float, nargs=2, default=[0.996, 1.0])
    ap.add_argument("--vicreg", type=float, default=0.0, help="variance hinge on predictor car-patch outputs")
    ap.add_argument("--ema-fixed", type=float, default=None, help="constant EMA momentum instead of the ramp")
    ap.add_argument("--fvar", type=float, default=0.0, help="Frame-variance hinge coefficient (position-centred)")
    ap.add_argument("--fvar-gamma", type=float, default=0.5, help="Frame-variance hinge target std")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-diag", type=int, default=2048)
    ap.add_argument("--readout", action="store_true", help="full Pose readout (ticket 01) at the end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    out = OUT / f"jepa_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    dev = C.pick_device(a.device)
    ds = open_dataset("data/step3-v1", check=False)
    tr_es, va_es = ds.split("train"), ds.split("val")
    sampler = C.FrameSampler(tr_es, a.seed, 11)
    mrng = np.random.default_rng([a.seed, 12])

    enc = Encoder(a.width, a.depth, a.heads, a.out_dim).to(dev)
    pred = Predictor(a.out_dim, a.pred_width, a.pred_depth, a.heads).to(dev)
    tgt = Encoder(a.width, a.depth, a.heads, a.out_dim).to(dev)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"device {dev}; encoder {n_enc:,} params, predictor {n_pred:,}", flush=True)

    decay, no_decay = [], []
    for m in (enc, pred):
        for n, p in m.named_parameters():
            (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    # fixed diagnostic Frames: val, first n_diag of the Step 4 eval sample (ridge fit on half, scored on half)
    dframes = cc.eval_frames(va_es, cc.N_FRAMES)[: a.n_diag]
    rng_d = np.random.default_rng(3)
    dframes = [dframes[i] for i in rng_d.permutation(len(dframes))]
    d_u8 = PV.render(va_es, dframes)
    _, d_st, _ = PV.labels(va_es, dframes)
    d_bg = np.stack([cc.background(va_es.track, va_es.render_params(e)) for e, _ in dframes])
    d_car = car_patches(cc.car_region(d_u8, d_bg))

    logf = open(out / "log.csv", "w", newline="")
    log = csv.writer(logf)
    log.writerow(["step", "loss", "loss_car", "loss_bg", "pred_std", "lr", "ema", "ctx_tokens", "car_hidden",
                  "frames_per_s", "fvar_hinge"])
    diagf = open(out / "diag.jsonl", "w")
    t0 = time.time()
    t_last, s_last = t0, 0
    acc = {"loss": 0.0, "car": 0.0, "bg": 0.0, "ncar": 0.0, "nbg": 0.0, "pstd": 0.0, "hid": 0.0, "k": 0.0, "fv": 0.0, "n": 0}
    t_data = 0.0
    for s in range(a.steps + 1):
        if s % a.eval_every == 0 or s == a.steps:
            tgt.eval()
            d = diagnostics(encode_all(tgt, d_u8, dev), d_car, d_st, len(dframes) // 2)
            d.update(step=s, minutes=(time.time() - t0) / 60)
            diagf.write(json.dumps(d) + "\n")
            diagf.flush()
            print(f"[{s:6d}] {d['minutes']:5.1f} min  gstd {d['gstd_min']:.2f}/{d['gstd_med']:.2f}  "
                  f"afstd {d['afstd_mean']:.3f}  contrast {d['car_contrast']:.2f}  rank {d['rank']:.1f}/"
                  f"{d['rank_poscentred']:.1f}  ridge {d['ridge_pos_p95']:.2f} m {d['ridge_head_p95']:.0f}°", flush=True)
        if s == a.steps:
            break
        td = time.time()
        frames, carpx = sampler.batch(a.batch)
        car = car_patches(carpx)
        ctx_idx, tgt_idx, _biased, hidden = sample_masks(car, mrng, a.car, a.p_car)
        t_data += time.time() - td
        B, M, T = tgt_idx.shape
        x = C.to_x(torch.from_numpy(frames).to(dev))
        ci = torch.from_numpy(ctx_idx).to(dev)
        ti = torch.from_numpy(tgt_idx).to(dev)
        with torch.no_grad():
            h = target_features(tgt, x)  # (B, 64, D)
            ht = torch.gather(h, 1, ti.reshape(B, M * T, 1).expand(-1, -1, h.shape[-1])).reshape(B * M, T, -1)
        z = enc(x, ci)  # (B, K, D)
        K = z.shape[1]
        zp = pred(z.repeat_interleave(M, 0), ci.repeat_interleave(M, 0), ti.reshape(B * M, T))
        err = ((zp - ht) ** 2).mean(-1)  # (B*M, T)
        loss = err.mean()
        is_car = torch.from_numpy(np.take_along_axis(car, tgt_idx.reshape(B, -1), 1).reshape(B * M, T)).to(dev)
        total = loss
        fv = torch.zeros((), device=dev)
        if a.fvar > 0:
            # std of each channel across the batch at each grid position, over the Frames whose context holds it
            zn = F.layer_norm(z, (z.shape[-1],))
            grid = torch.zeros(B, G * G, zn.shape[-1], device=dev).scatter(1, ci[..., None].expand_as(zn), zn)
            cnt = torch.zeros(B, G * G, 1, device=dev).scatter(1, ci[..., None], 1.0)
            n_p = cnt.sum(0)  # (64, 1)
            ok = (n_p[:, 0] >= 8)
            mu = grid.sum(0) / n_p.clamp(min=1)
            var = (((grid - mu) ** 2) * cnt).sum(0) / (n_p - 1).clamp(min=1)
            fv = F.relu(a.fvar_gamma - torch.sqrt(var[ok] + 1e-4)).mean()
            total = total + a.fvar * fv
        if a.vicreg > 0 and is_car.any():
            zc = zp[is_car]
            total = loss + a.vicreg * F.relu(1 - torch.sqrt(zc.var(0) + 1e-4)).mean()
        lr = lr_at(s, a.steps, a.lr)
        wd = cosine(s, a.steps, 0.04, 0.4)
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr
            if gi == 0:
                g["weight_decay"] = wd
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step()
        m = a.ema_fixed if a.ema_fixed is not None else a.ema[0] + (a.ema[1] - a.ema[0]) * s / a.steps
        with torch.no_grad():
            for pt, pe in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(m).add_(pe.detach(), alpha=1 - m)
        with torch.no_grad():
            ic = is_car.float()
            acc["loss"] += float(loss)
            acc["car"] += float((err * ic).sum())
            acc["ncar"] += float(ic.sum())
            acc["bg"] += float((err * (1 - ic)).sum())
            acc["nbg"] += float((1 - ic).sum())
            acc["pstd"] += float(zp.reshape(-1, zp.shape[-1]).std(0).mean())
            acc["hid"] += float(hidden.mean())
            acc["k"] += K
            acc["fv"] += float(fv)
            acc["n"] += 1
        if (s + 1) % 100 == 0:
            n = acc["n"]
            fps = (s + 1 - s_last) * a.batch / (time.time() - t_last)
            row = [s + 1, acc["loss"] / n, acc["car"] / max(acc["ncar"], 1), acc["bg"] / max(acc["nbg"], 1),
                   acc["pstd"] / n, lr, m, acc["k"] / n, acc["hid"] / n, fps, acc["fv"] / n]
            log.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
            logf.flush()
            if (s + 1) % 500 == 0:
                print(f"  step {s + 1}: loss {row[1]:.4f} (car {row[2]:.4f}, bg {row[3]:.4f})  pstd {row[4]:.3f}  "
                      f"ctx {row[7]:.1f}  car hidden {row[8]:.2f}  {fps:.0f} Frames/s (data {t_data / (time.time() - t0):.0%})",
                      flush=True)
            acc = {k: 0.0 for k in acc}
            acc["n"] = 0
            t_last, s_last = time.time(), s + 1
    train_min = (time.time() - t0) / 60
    torch.save({"target": tgt.state_dict(), "args": vars(a)}, out / "target.pt")
    res = {"name": a.name, "args": vars(a), "encoder_params": n_enc, "predictor_params": n_pred,
           "train_minutes": train_min, "frames_per_s": a.steps * a.batch / (train_min * 60)}
    if a.readout:
        res["readout"] = pose_readout(tgt, tr_es, va_es, dev)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))


def pose_readout(tgt: Encoder, tr_es, va_es, dev) -> dict:
    """Ticket 01's Pose readout on the flattened target grid: MLP (gate) + ridge (reported)."""
    t0 = time.time()
    tr = PV.sample_frames(tr_es, 65536, 7)
    Ytr, sttr, _ = PV.labels(tr_es, tr)
    Xtr = encode_all(tgt, PV.render(tr_es, tr), dev).reshape(len(tr), -1)
    ev = cc.eval_frames(va_es, cc.N_FRAMES)
    _, stev, _ = PV.labels(va_es, ev)
    Xev = {"clean": encode_all(tgt, PV.render(va_es, ev), dev).reshape(len(ev), -1)}
    for p in C.COLOUR_PROBES:
        Xev[p] = encode_all(tgt, PV.render(va_es, ev, p), dev).reshape(len(ev), -1)
    eps = np.array([e for e, _ in tr])
    hold = np.isin(eps, np.random.default_rng(0).choice(len(tr_es), len(tr_es) // 5, replace=False))
    sel = {}
    for lam in PV.LAMBDAS:
        r = PV.Ridge().fit(Xtr[~hold], Ytr[~hold], lam)
        pos, head = PV.errors(r.predict(Xtr[hold]), sttr[hold])
        sel[lam] = float(np.percentile(pos, 95)) / 0.3 + float(np.percentile(head, 95)) / 10
    lam = min(sel, key=sel.get)
    ridge = PV.Ridge().fit(Xtr, Ytr, lam)
    res = {"lambda": lam, "ridge": {}, "mlp": {}}
    for c, X in Xev.items():
        res["ridge"][c] = PV.stats(*PV.errors(ridge.predict(X), stev))
    conds = list(Xev)
    for c, P in zip(conds, PV.mlp_probe(Xtr, Ytr, [Xev[c] for c in conds])):
        res["mlp"][c] = PV.stats(*PV.errors(P, stev))
    res["seconds"] = time.time() - t0
    for tier in ("mlp", "ridge"):
        s = res[tier]["clean"]
        print(f"{tier:5s} clean: pos p50 {s['pos_p50']:.3f} p95 {s['pos_p95']:.3f} max {s['pos_max']:.2f} m   "
              f"head p50 {s['head_p50']:.1f} p95 {s['head_p95']:.1f} max {s['head_max']:.0f}°")
    return res


if __name__ == "__main__":
    main()
