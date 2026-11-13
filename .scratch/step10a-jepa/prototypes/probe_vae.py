"""Ticket 01 prototype: what a linear probe reads from frozen `vae-v1` Latents.

    uv run python .scratch/step10a-jepa/prototypes/probe_vae.py [--train 65536] [--mlp]

Probe: ridge regression from the flattened normalised Latent (16·8·8 = 1024 values, plus bias) to
(x, y, cos θ, sin θ); heading decoded with atan2. Inputs standardised per feature on the probe's
train Frames. λ picked on a 20 % held-out part of the train Frames, then refit on all of them.
Train Frames: per-episode quotas from the train Split (stream [seed, 7]); eval Frames: the Step 4
check's 4096 val Frames (`carcheck.eval_frames`, stream [seed, 5]), clean and per colour Probe
(probe fitted on clean train Frames only). Errors vs the true State.
Reference: an MLP probe (1024 → 512 → 512 → 4) on the same Frames, to see how much is there non-linearly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from toycar import State
from wam import carcheck as cc
from wam import compressor as C
from toycar.dataset import open_dataset

OUT = Path(__file__).parent / "out"
LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def sample_frames(es, n: int, stream: int) -> list[tuple[int, int]]:
    E = len(es)
    out = []
    for i in range(E):
        q = n // E + (1 if i < n % E else 0)
        if q == 0:
            continue
        rng = np.random.default_rng([int(es.arrays["seed"][i]), stream])
        ticks = np.sort(rng.choice(int(es.arrays["length"][i]) + 1, size=q, replace=False))
        out += [(i, int(t)) for t in ticks]
    return out


def render(es, frames, probe=None) -> np.ndarray:
    return np.stack([es._render(e, t, es.render_params(e, probe)) for e, t in frames])


def labels(es, frames) -> tuple[np.ndarray, np.ndarray, list]:
    st = np.array([es.arrays["states"][e, t] for e, t in frames], np.float64)  # x, y, theta, v
    y = np.stack([st[:, 0], st[:, 1], np.cos(st[:, 2]), np.sin(st[:, 2])], 1)
    return y, st, [int(es.arrays["branch"][e, t]) for e, t in frames]


def encode(comp, frames_u8: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(frames_u8), 2048):
        out.append(comp.encode(frames_u8[i : i + 2048]).reshape(-1, 1024).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


class Ridge:
    def fit(self, X, Y, lam):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        A = Xs.T @ Xs
        A[np.diag_indices(len(A) - 1)] += lam  # no penalty on the bias
        self.W = np.linalg.solve(A, Xs.T @ Y)
        return self

    def predict(self, X):
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.W


def errors(P, st):
    pos = np.hypot(P[:, 0] - st[:, 0], P[:, 1] - st[:, 1])
    th = np.arctan2(P[:, 3], P[:, 2])
    head = np.abs(np.degrees((th - st[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    return pos, head


def stats(pos, head):
    q = lambda a, p: float(np.percentile(a, p))
    return {"pos_p50": q(pos, 50), "pos_p95": q(pos, 95), "pos_max": float(pos.max()),
            "head_p50": q(head, 50), "head_p95": q(head, 95), "head_max": float(head.max())}


def slices(es, frames, st, branch):
    track = es.track
    off = np.array([min(track.route("outer").project(s[0], s[1]).distance_m,
                        track.route("inner").project(s[0], s[1]).distance_m) for s in st])
    return {"all": np.ones(len(st), bool), "section": np.array(branch) != -1,
            "fast": st[:, 3] >= cc.FAST_MPS, "kerb": off >= cc.KERB_M}


def fmt(name, s, n):
    return (f"  {name:14s} n={n:5d}  pos p50 {s['pos_p50']:.3f}  p95 {s['pos_p95']:.3f}  max {s['pos_max']:.3f} m"
            f"   head p50 {s['head_p50']:5.1f}  p95 {s['head_p95']:5.1f}  max {s['head_max']:6.1f}°")


def mlp_probe(Xtr, Ytr, Xev, seed=0, steps=4000):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0)
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((Ytr - ymu) / ysd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, 4))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(xt), (256,), generator=g)
        loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return [net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy() * ysd + ymu for X in Xev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=65536)
    ap.add_argument("--mlp", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ds = open_dataset("data/step3-v1", check=False)
    tr_es, va_es = ds.split("train"), ds.split("val")
    comp = C.load("vae-v1")
    print(f"device {comp.device}", flush=True)

    tr = sample_frames(tr_es, a.train, 7)
    Ytr, sttr, _ = labels(tr_es, tr)
    Xtr = encode(comp, render(tr_es, tr))
    print(f"train: {len(tr)} Frames encoded ({time.time() - t0:.0f} s)", flush=True)

    ev = cc.eval_frames(va_es, cc.N_FRAMES)
    Yev, stev, br = labels(va_es, ev)
    Xev = {"clean": encode(comp, render(va_es, ev))}
    for p in C.COLOUR_PROBES:
        Xev[p] = encode(comp, render(va_es, ev, p))
    print(f"eval: {len(ev)} val Frames × {len(Xev)} conditions ({time.time() - t0:.0f} s)", flush=True)

    # λ on a held-out 20 % of the train Frames (by episode, so neighbouring Ticks don't leak)
    eps = np.array([e for e, _ in tr])
    hold = np.isin(eps, np.random.default_rng(0).choice(len(tr_es), len(tr_es) // 5, replace=False))
    sel = {}
    for lam in LAMBDAS:
        r = Ridge().fit(Xtr[~hold], Ytr[~hold], lam)
        pos, head = errors(r.predict(Xtr[hold]), sttr[hold])
        sel[lam] = (float(np.percentile(pos, 95)), float(np.percentile(head, 95)))
        print(f"  λ={lam:g}: held-out pos p95 {sel[lam][0]:.3f} m, head p95 {sel[lam][1]:.1f}°", flush=True)
    lam = min(sel, key=lambda k: sel[k][0] / 0.3 + sel[k][1] / 10.0)
    ridge = Ridge().fit(Xtr, Ytr, lam)
    print(f"chosen λ = {lam:g}; train fit pos p95 {np.percentile(errors(ridge.predict(Xtr), sttr)[0], 95):.3f} m")

    res: dict = {"train_frames": len(tr), "lambda": lam, "lambda_sweep": {str(k): v for k, v in sel.items()},
                 "ridge": {}, "mlp": {}}
    sl = slices(va_es, ev, stev, br)
    print(f"\nLINEAR (ridge, λ={lam:g}) on vae-v1 Latents")
    for cond, X in Xev.items():
        pos, head = errors(ridge.predict(X), stev)
        res["ridge"][cond] = {}
        for k, m in sl.items():
            if cond != "clean" and k != "all":
                continue
            s = stats(pos[m], head[m])
            res["ridge"][cond][k] = s
            print(fmt(f"{cond}/{k}", s, int(m.sum())), flush=True)
    # baseline: predict the mean (how big is the spread at all)
    pos0, head0 = errors(np.repeat(Ytr.mean(0, keepdims=True), len(stev), 0), stev)
    res["mean_baseline"] = stats(pos0, head0)
    print(fmt("mean-baseline", res["mean_baseline"], len(stev)))

    if a.mlp:
        conds = list(Xev)
        preds = mlp_probe(Xtr, Ytr, [Xev[c] for c in conds])
        print("\nMLP probe (reference, non-linear)")
        for c, P in zip(conds, preds):
            pos, head = errors(P, stev)
            res["mlp"][c] = stats(pos, head)
            print(fmt(c, res["mlp"][c], len(stev)), flush=True)

    res["seconds"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / f"probe_vae_{a.train}.json").write_text(json.dumps(res, indent=1))
    print(f"\ndone in {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
