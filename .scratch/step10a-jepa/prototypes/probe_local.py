"""Ticket 01 prototype, part 2: a *local* probe on frozen `vae-v1` Latents.

    uv run python .scratch/step10a-jepa/prototypes/probe_local.py

The plain ridge probe (probe_vae.py) can't read position: the flattened grid encodes *where* by
which cell is active, and a single linear map is too coarse (pos p95 1.5 m) though an MLP reads it
(0.33 m). This probe keeps every per-cell map linear and shared across cells (translation-equivariant)
and adds only a softmax over cells to locate the car:

    a_c   = softmax_c(w · f_c + b)                  # where is the car (one linear score per cell)
    x, y  = Σ_c a_c (centre_c + U f_c + u)          # cell centre + linear sub-cell offset
    cos, sin = Σ_c a_c (V f_c + v)                  # linear heading readout at the car's cell(s)

f_c = the C channels of cell c (k=1), or of its k×k neighbourhood (zero-padded; k=3).
Trained with AdamW on (position MSE in metres + heading-vector MSE). Same Frames as probe_vae.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from probe_vae import encode, errors, fmt, labels, render, sample_frames, slices, stats  # noqa: E402

from wam import carcheck as cc  # noqa: E402
from wam import compressor as C  # noqa: E402
from toycar.dataset import open_dataset  # noqa: E402

OUT = Path(__file__).parent / "out"
M_PER_CELL = 40.0 / 8  # 5 m


class LocalProbe(torch.nn.Module):
    def __init__(self, C_: int, k: int, centres: torch.Tensor):
        super().__init__()
        self.k = k
        d = C_ * k * k
        self.score = torch.nn.Linear(d, 1)
        self.off = torch.nn.Linear(d, 2)
        self.head = torch.nn.Linear(d, 2)
        self.register_buffer("centres", centres)  # (64, 2) metres

    def feats(self, z):  # z (B, C, 8, 8) → (B, 64, C·k·k)
        if self.k == 1:
            return z.flatten(2).transpose(1, 2)
        return F.unfold(z, self.k, padding=self.k // 2).transpose(1, 2)

    def forward(self, z):
        f = self.feats(z)
        a = torch.softmax(self.score(f).squeeze(-1), dim=1)  # (B, 64)
        pos = (a[..., None] * (self.centres + self.off(f))).sum(1)
        hv = (a[..., None] * self.head(f)).sum(1)
        return torch.cat([pos, hv], 1), a


# Frame covers [0, 40] m in x (columns) and y (rows, row 0 = small y) — toycar.render._layout
r_, c_ = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
CENTRES = np.stack([(c_.ravel() + 0.5) * M_PER_CELL, (r_.ravel() + 0.5) * M_PER_CELL], 1)  # (64, 2), index r·8+c


def train_probe(Xtr, Ytr, centres, k, steps=6000, seed=0, dev="cpu"):
    torch.manual_seed(seed)
    m = LocalProbe(Xtr.shape[1], k, centres).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(Ytr, dtype=torch.float32)
    for s in range(steps):
        i = torch.randint(len(X), (256,), generator=g)
        p, _ = m(X[i].to(dev))
        y = Y[i].to(dev)
        loss = ((p[:, :2] - y[:, :2]) ** 2).sum(1).mean() + 10 * ((p[:, 2:] - y[:, 2:]) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m.cpu().eval()


def predict(m, X):
    with torch.no_grad():
        return torch.cat([m(torch.tensor(X[i:i + 4096], dtype=torch.float32))[0]
                          for i in range(0, len(X), 4096)]).numpy()


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    t0 = time.time()
    ds = open_dataset("data/step3-v1", check=False)
    tr_es, va_es = ds.split("train"), ds.split("val")
    comp = C.load("vae-v1")
    tr = sample_frames(tr_es, n_train, 7)
    Ytr, sttr, _ = labels(tr_es, tr)
    Xtr = encode(comp, render(tr_es, tr)).reshape(-1, 16, 8, 8)
    ev = cc.eval_frames(va_es, cc.N_FRAMES)
    Yev, stev, br = labels(va_es, ev)
    Xev = {"clean": encode(comp, render(va_es, ev)).reshape(-1, 16, 8, 8)}
    for p in C.COLOUR_PROBES:
        Xev[p] = encode(comp, render(va_es, ev, p)).reshape(-1, 16, 8, 8)
    print(f"encoded {len(tr)} train + {len(ev)}×{len(Xev)} eval Frames ({time.time() - t0:.0f} s)", flush=True)

    res = {"train_frames": len(tr)}
    sl = slices(va_es, ev, stev, br)
    for k in (1, 3):
        centres = torch.tensor(CENTRES, dtype=torch.float32)
        t1 = time.time()
        m = train_probe(Xtr, Ytr, centres, k)
        print(f"\nLOCAL probe k={k} ({sum(p.numel() for p in m.parameters())} params, {time.time() - t1:.0f} s)")
        res[f"k{k}"] = {}
        for cond, X in Xev.items():
            pos, head = errors(predict(m, X), stev)
            for name, mask in sl.items():
                if cond != "clean" and name != "all":
                    continue
                s = stats(pos[mask], head[mask])
                res[f"k{k}"][f"{cond}/{name}"] = s
                print(fmt(f"{cond}/{name}", s, int(mask.sum())), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"probe_local_{n_train}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
