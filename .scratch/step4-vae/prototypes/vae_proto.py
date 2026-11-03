"""PROTOTYPE — throwaway. Ticket 03 "Architecture and training objective".

Trains the ticket 03 VAE variants for a fixed number of steps and scores each with the ticket 04
car-visibility check (clean and noisy latents, plus the four colour Probes, report-only).

Run from the repo root (torch and matplotlib are not project dependencies yet):

    uv run --group viz --with torch==2.15.0 python .scratch/step4-vae/prototypes/vae_proto.py --run A
    uv run --group viz --with torch==2.15.0 python .scratch/step4-vae/prototypes/vae_proto.py --run all
    # smoke test:  --run A --steps 50 --stats-frames 1024 --eval-frames 256

Runs (ticket 03, Q5):   A: KL 0, λ 0   B: KL 0, λ 30   C: KL 1e-4, λ 30   D: KL 1e-3, λ 30
                        E: bigger net (2 residual blocks per stage), KL from --kl, λ 30
Writes prototypes/out/vae_<run>/: log.csv, metrics.json, contact.png, model.pt (gitignored size).
Ctrl-C stops the current run; finished runs keep their results.
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

from pose_estimator import FULL, PX, THRESH, estimate, eval_frames, psnr, summarise  # same directory
from toycar import State, render
from toycar.dataset import PROBES, open_dataset

OUT = Path(__file__).parent / "out"
DATA = "data/step3-v1"
RUNS = {
    "A": dict(kl=0.0, lam=0.0, blocks=1),
    "B": dict(kl=0.0, lam=30.0, blocks=1),
    "C": dict(kl=1e-4, lam=30.0, blocks=1),
    "D": dict(kl=1e-3, lam=30.0, blocks=1),
    "E": dict(kl=None, lam=30.0, blocks=2),
}
COLOUR_PROBES = [n for n, p in PROBES.items() if p.track_params is None]
NOISE_STD = 0.1  # noisy-latent test, in normalised units
Z = 16


# -- model (ticket 03, Q3; lean first stage from ticket 01) ---------------------------------
def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class Res(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.n1, self.c1 = gn(c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = gn(c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class VAE(nn.Module):
    """64×64×3 → 8×8×16 (mean, log-variance) → 64×64×3. Channels 32 / 64 / 128 / 128.
    No residual block at 64×64 (lean). `blocks` residual blocks per lower stage."""

    def __init__(self, blocks: int = 1) -> None:
        super().__init__()
        res = lambda c: [Res(c) for _ in range(blocks)]  # noqa: E731
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), *res(64),      # 32×32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), *res(128),    # 16×16
            nn.Conv2d(128, 128, 3, stride=2, padding=1), *res(128),   # 8×8
            gn(128), nn.SiLU(), nn.Conv2d(128, 2 * Z, 1),
        )
        up = lambda a, b: [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1)]  # noqa: E731
        self.dec = nn.Sequential(
            nn.Conv2d(Z, 128, 3, padding=1), *res(128),               # 8×8
            *up(128, 128), *res(128),                                 # 16×16
            *up(128, 64), *res(64),                                   # 32×32
            *up(64, 32),                                              # 64×64
            gn(32), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def decode(self, z):
        return self.dec(z)


def to_x(u8: np.ndarray, dev) -> torch.Tensor:
    """(B, 64, 64, 3) uint8 → (B, 3, 64, 64) float in [−1, 1] on dev."""
    return torch.from_numpy(u8).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def to_u8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


# -- data -----------------------------------------------------------------------------------
_EMPTY = State(-100.0, -100.0, 0.0, 0.0)


def grow(mask: np.ndarray) -> np.ndarray:
    """Grow a (…, 64, 64) bool mask by 1 pixel in every direction (incl. diagonals)."""
    g = mask.copy()
    g[..., 1:, :] |= mask[..., :-1, :]; g[..., :-1, :] |= mask[..., 1:, :]
    h = g.copy()
    h[..., :, 1:] |= g[..., :, :-1]; h[..., :, :-1] |= g[..., :, 1:]
    return h


class Sampler:
    """Random train Frames with their car-region masks (ticket 03, Q2)."""

    def __init__(self, es, seed: int) -> None:
        self.es, self.rng = es, np.random.default_rng(seed)
        self.bg = {}

    def background(self, e: int, probe: str | None = None) -> np.ndarray:
        key = (e, probe)
        if key not in self.bg:
            self.bg[key] = render(_EMPTY, self.es.track, self.es.render_params(e, probe))
        return self.bg[key]

    def batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        es = self.es
        frames = np.empty((n, 64, 64, 3), np.uint8)
        bgs = np.empty_like(frames)
        for j in range(n):
            e = int(self.rng.integers(len(es)))
            t = int(self.rng.integers(int(es.arrays["length"][e]) + 1))
            frames[j] = es._render(e, t, es.render_params(e))
            bgs[j] = self.background(e)
        return frames, grow((frames != bgs).any(-1))


# -- training -------------------------------------------------------------------------------
def train(cfg: dict, steps: int, batch: int, seed: int, dev, log_path: Path) -> tuple[VAE, dict]:
    torch.manual_seed(seed)
    model = VAE(cfg["blocks"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ds = open_dataset(DATA, check=False)
    sampler = Sampler(ds.split("train"), seed)
    kl_w, lam = cfg["kl"], cfg["lam"]
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "loss", "rec", "kl", "lr", "elapsed_s"])
        acc = np.zeros(3)
        for step in range(1, steps + 1):
            frames, mask = sampler.batch(batch)
            x = to_x(frames, dev)
            w = 1.0 + lam * torch.from_numpy(mask).to(dev).float()[:, None]
            mu, logvar = model.encode(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if kl_w > 0 else mu
            rec = (w * (model.decode(z) - x).abs()).sum() / (w.sum() * 3)  # weighted mean L1 per value
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()
            loss = rec + kl_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc += [loss.item(), rec.item(), kl.item()]
            if step % 100 == 0 or step == steps:
                n = 100 if step % 100 == 0 else step % 100
                el = time.time() - t0
                lr = sched.get_last_lr()[0]
                log.writerow([step, *(acc / n).round(6), f"{lr:.2e}", round(el, 1)])
                f.flush()
                eta = el / step * (steps - step)
                print(f"  step {step:6d}/{steps}  loss {acc[0] / n:.4f}  rec {acc[1] / n:.4f}  kl {acc[2] / n:8.3f}  "
                      f"lr {lr:.1e}  {step * batch / el:5.0f} Frames/s  ETA {eta / 60:5.1f} min", flush=True)
                acc[:] = 0
    return model, {"train_s": round(time.time() - t0, 1), "params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def latent_stats(model: VAE, n: int, dev, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of mean latents over n random train Frames (ticket 03, Q4)."""
    sampler = Sampler(open_dataset(DATA, check=False).split("train"), seed + 1)
    s1 = torch.zeros(Z, device=dev, dtype=torch.float32)
    s2 = torch.zeros_like(s1)
    cnt = 0
    for i in range(0, n, 256):
        frames, _ = sampler.batch(min(256, n - i))
        mu, _ = model.encode(to_x(frames, dev))
        s1 += mu.sum(dim=(0, 2, 3)); s2 += mu.pow(2).sum(dim=(0, 2, 3))
        cnt += mu.shape[0] * mu.shape[2] * mu.shape[3]
    mean = s1 / cnt
    std = (s2 / cnt - mean.pow(2)).clamp_min(1e-12).sqrt()
    return mean, std


# -- ticket 04 check ------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model, frames_u8, dev, mean, std, noise: float, gen: torch.Generator) -> np.ndarray:
    out = np.empty_like(frames_u8)
    m, s = mean.view(1, Z, 1, 1), std.view(1, Z, 1, 1)
    for i in range(0, len(frames_u8), 256):
        mu, _ = model.encode(to_x(frames_u8[i : i + 256], dev))
        zn = (mu - m) / s
        if noise:
            zn = zn + noise * torch.randn(zn.shape, generator=gen).to(dev)
        out[i : i + 256] = to_u8(model.decode(zn * s + m))
    return out


def check(model, es, frames, dev, mean, std, label, probe=None, noise=0.0):
    """Ticket 04 check; returns (summary, per-frame rows, originals, reconstructions)."""
    track = es.track
    rps = [es.render_params(e, probe) for e, _ in frames]
    states = [State(*map(float, es.arrays["states"][e, t])) for e, t in frames]
    orig = np.stack([render(s, track, rp) for s, rp in zip(states, rps)])
    bgs = np.stack([render(_EMPTY, track, rp) for rp in rps])
    rec = reconstruct(model, orig, dev, mean, std, noise, torch.Generator().manual_seed(7))
    rows = []
    for i, ((e, t), s, rp) in enumerate(zip(frames, states, rps)):
        po = estimate(orig[i], bgs[i], rp)
        pr = estimate(rec[i], bgs[i], rp, centre=po[:2])
        region = grow((orig[i] != bgs[i]).any(-1))
        found = pr is not None
        pos = math.hypot(pr[0] - po[0], pr[1] - po[1]) if found else math.inf
        head = abs(math.degrees((pr[2] - po[2] + math.pi) % (2 * math.pi) - math.pi)) if found and pr[2] is not None else 180.0
        off = min(track.route("outer").project(s.x, s.y).distance_m, track.route("inner").project(s.x, s.y).distance_m)
        truth = (math.hypot(pr[0] - s.x, pr[1] - s.y) if found else math.inf,
                 abs(math.degrees((pr[2] - s.theta + math.pi) % (2 * math.pi) - math.pi)) if found and pr[2] is not None else 180.0)
        rows.append(dict(found=found, pos=pos, head=head, pos_true=truth[0], head_true=truth[1],
                         psnr=psnr(orig[i], rec[i]), psnr_car=psnr(orig[i], rec[i], region), psnr_bg=psnr(orig[i], rec[i], ~region),
                         section=bool(es.arrays["branch"][e, t] != -1), fast=s.v >= 8.0, kerb=off >= 0.5, po=po, pr=pr))
    summary = summarise(rows, label)
    fin = [r for r in rows if r["found"]]
    summary["all"]["vs_true_pos_p95"] = float(np.percentile([r["pos_true"] for r in fin], 95)) if fin else None
    summary["all"]["vs_true_head_p95"] = float(np.percentile([r["head_true"] for r in rows], 95))
    return summary, rows, orig, rec


# -- contact sheet (ticket 04, Q7) ----------------------------------------------------------
def contact_sheet(path: Path, title: str, clean, probes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, orig, rec = clean
    rng = np.random.default_rng(0)
    n = len(rows)
    pick = [("random", int(i)) for i in rng.choice(n, 8, replace=False)]
    pick += [("worst pos", int(i)) for i in np.argsort([-r["pos"] for r in rows])[:4]]
    pick += [("worst head", int(i)) for i in np.argsort([-r["head"] for r in rows])[:4]]
    for k in ("section", "fast", "kerb"):
        idx = [i for i, r in enumerate(rows) if r[k]]
        pick += [(k, int(i)) for i in rng.choice(idx, min(2, len(idx)), replace=False)]
    entries = [(lab, rows[i], orig[i], rec[i]) for lab, i in pick]
    for name, (prow, porig, prec) in probes.items():
        entries += [(name, prow[int(i)], porig[int(i)], prec[int(i)]) for i in rng.choice(len(prow), 2, replace=False)]

    per_row = 5
    nr = math.ceil(len(entries) / per_row)
    fig, axes = plt.subplots(nr, per_row * 4, figsize=(per_row * 4 * 0.9, nr * 1.25))
    for ax in axes.ravel():
        ax.axis("off")
    for j, (lab, r, o, c) in enumerate(entries):
        a = axes[j // per_row, (j % per_row) * 4 : (j % per_row) * 4 + 4]
        cx, cy = r["po"][0] * PX, r["po"][1] * PX
        h = 6
        x0, y0 = int(min(max(round(cx) - h, 0), 64 - 2 * h)), int(min(max(round(cy) - h, 0), 64 - 2 * h))
        for k, img in enumerate([o, c, o[y0 : y0 + 2 * h, x0 : x0 + 2 * h], c[y0 : y0 + 2 * h, x0 : x0 + 2 * h]]):
            a[k].imshow(img, interpolation="nearest")
        for k, p in ((2, r["po"]), (3, r["pr"])):
            if p is not None and p[2] is not None:
                px, py = p[0] * PX - x0 - 0.5, p[1] * PX - y0 - 0.5
                a[k].arrow(px, py, 3 * math.cos(p[2]), 3 * math.sin(p[2]), color="cyan", width=0.25, head_width=1.2)
        a[0].set_title(lab, fontsize=6, loc="left")
        a[1].set_title(f"Δp {r['pos']:.2f} m  Δθ {r['head']:.0f}°", fontsize=6, loc="left",
                       color="red" if r["pos"] > THRESH["pos_max"] or r["head"] > THRESH["head_max"] else "black")
    fig.suptitle(title + "   (each entry: original | reconstruction | zoom original | zoom reconstruction)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -- one run --------------------------------------------------------------------------------
def jsonable(s):
    return {k: {kk: (None if isinstance(vv, float) and not math.isfinite(vv) else
                     (bool(vv) if isinstance(vv, (bool, np.bool_)) else float(vv) if isinstance(vv, (float, np.floating)) else vv))
                for kk, vv in v.items()} for k, v in s.items()}


def run(name: str, a) -> dict:
    cfg = dict(RUNS[name])
    if name == "E":
        if a.kl is None:
            raise SystemExit("--run E needs --kl (the KL weight of the best of C/D)")
        cfg["kl"] = a.kl
    dev = torch.device("mps" if torch.backends.mps.is_available() and not a.cpu else "cpu")
    out = OUT / f"vae_{name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== run {name}: {cfg}, {a.steps} steps × batch {a.batch} on {dev} ===", flush=True)
    model, info = train(cfg, a.steps, a.batch, a.seed, dev, out / "log.csv")
    model.eval()
    print(f"  trained in {info['train_s'] / 60:.1f} min, {info['params'] / 1e6:.2f} M params; latent stats …", flush=True)
    mean, std = latent_stats(model, a.stats_frames, dev, a.seed)
    torch.save({"config": cfg, "state_dict": model.state_dict(), "latent_mean": mean.cpu(), "latent_std": std.cpu()}, out / "model.pt")

    es = open_dataset(DATA, check=False).split("val")
    frames = eval_frames(es, a.eval_frames)
    s_clean, rows, orig, rec = check(model, es, frames, dev, mean, std, f"run {name} — clean latents")
    s_noisy, *_ = check(model, es, frames, dev, mean, std, f"run {name} — noisy latents (+{NOISE_STD}·N)", noise=NOISE_STD)
    probes, probe_sheets = {}, {}
    for p in COLOUR_PROBES:
        s, prow, porig, prec = check(model, es, frames, dev, mean, std, f"run {name} — probe {p} (report only)", probe=p)
        probes[p], probe_sheets[p] = s, (prow, porig, prec)
    contact_sheet(out / "contact.png", f"run {name}: {cfg}", (rows, orig, rec), probe_sheets)
    metrics = {
        "run": name, "config": cfg, "steps": a.steps, "batch": a.batch, "seed": a.seed, **info,
        "latent_mean": [round(float(v), 4) for v in mean.cpu()], "latent_std": [round(float(v), 4) for v in std.cpu()],
        "clean": jsonable(s_clean), "noisy": jsonable(s_noisy), "probes": {k: jsonable(v) for k, v in probes.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"  latent std per channel: {' '.join(f'{v:.3g}' for v in std.cpu())}")
    print(f"  → {out}  clean {'PASS' if s_clean['all']['pass'] else 'FAIL'}, noisy {'PASS' if s_noisy['all']['pass'] else 'FAIL'}", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="A, B, C, D, E, or 'all' (= A B C D)")
    ap.add_argument("--steps", type=int, default=6000, help="training steps (default ≈ 30 min at batch 128, about 430 Frames/s)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--kl", type=float, help="KL weight for run E")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-frames", type=int, default=65536)
    ap.add_argument("--eval-frames", type=int, default=4096)
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow)")
    a = ap.parse_args()
    names = ["A", "B", "C", "D"] if a.run == "all" else [a.run]
    if any(n not in RUNS for n in names):
        raise SystemExit(f"unknown run {a.run!r}")
    results = []
    for n in names:
        try:
            results.append(run(n, a))
        except KeyboardInterrupt:
            print(f"\nrun {n} interrupted; no results written for it", flush=True)
            break
    if len(results) > 1:
        print("\nsummary (clean / noisy):")
        for m in results:
            c, z = m["clean"]["all"], m["noisy"]["all"]
            print(f"  {m['run']}  pos p95 {c['pos_p95']:.3f}  head p95 {c['head_p95']:.1f}  head max {c['head_max']:.1f}  "
                  f"car PSNR {c['psnr_car']:.1f}  {'PASS' if c['pass'] else 'FAIL'} / {'PASS' if z['pass'] else 'FAIL'}")


if __name__ == "__main__":
    sys.exit(main())
