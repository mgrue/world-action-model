"""Ticket 04 prototype: the car-visibility check, validated on hand-made fake reconstructions.

    uv run python .scratch/step4-vae/prototypes/pose_estimator.py

Implements the ticket 04 decisions: colour-centroid pose estimator searched within ±4 m of the
original's estimate; pass/fail on reconstruction-vs-original pose differences over 4096 fixed val
Frames; PSNR diagnostics (whole Frame, car region = car mask grown by 1 px, background);
per-situation breakdown. Probes are report-only and not exercised here (no compressor yet).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from toycar import State, render
from toycar.dataset import open_dataset

PX = 64 / 40.0  # pixels per metre
SEARCH_M = 4.0  # search square half-width around the original's estimate
CUTOFF = 9.0  # summed |ΔRGB| below this is ignored
FULL = 60.0  # summed |ΔRGB| at which a pixel counts fully as car
N_FRAMES = 4096
THRESH = {"pos_p95": 0.3, "pos_max": 0.6, "head_p95": 10.0, "head_max": 30.0}


def estimate(frame, background, rp, centre=None):
    """(x, y, θ or None) in metres/rad, or None if no car. `centre` = (x, y) in metres limits the search."""
    f = frame.astype(np.float64)
    diff = np.abs(f - background.astype(np.float64)).sum(-1)
    w = np.clip(diff / FULL, 0, 1)
    w[diff < CUTOFF] = 0
    ys, xs = np.mgrid[0:64, 0:64] + 0.5
    if centre is not None:
        cx0, cy0 = centre[0] * PX, centre[1] * PX
        r = SEARCH_M * PX
        w[(np.abs(xs - cx0) > r) | (np.abs(ys - cy0) > r)] = 0
    if w.sum() < 1.0:
        return None
    db = np.linalg.norm(f - np.array(rp.body, float), axis=-1)
    dn = np.linalg.norm(f - np.array(rp.nose, float), axis=-1)
    wn = w * (dn < db)
    cx, cy = (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()
    if wn.sum() < 0.2:
        return cx / PX, cy / PX, None
    nx, ny = (wn * xs).sum() / wn.sum(), (wn * ys).sum() / wn.sum()
    return cx / PX, cy / PX, math.atan2(ny - cy, nx - cx)


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)


def eval_frames(es, n=N_FRAMES):
    """Fixed (episode, tick) sample: per-episode quotas, ticks drawn with stream [seed, 5]."""
    E = len(es)
    out = []
    for i in range(E):
        q = n // E + (1 if i < n % E else 0)
        rng = np.random.default_rng([int(es.arrays["seed"][i]), 5])
        ticks = np.sort(rng.choice(int(es.arrays["length"][i]) + 1, size=q, replace=False))
        out += [(i, int(t)) for t in ticks]
    return out


def evaluate(es, frames, recon_fn: Callable, label: str):
    track = es.track
    rows = []
    for e, t in frames:
        rp = es.render_params(e)
        s = State(*map(float, es.arrays["states"][e, t]))
        orig = render(s, track, rp)
        bg = render(State(-100, -100, 0, 0), track, rp)
        rec = recon_fn(orig, s, track, rp, bg)
        po = estimate(orig, bg, rp)
        pr = estimate(rec, bg, rp, centre=po[:2])
        mask = np.abs(orig.astype(int) - bg.astype(int)).sum(-1) > 0
        grown = mask.copy()
        grown[1:] |= mask[:-1]; grown[:-1] |= mask[1:]
        g2 = grown.copy()
        g2[:, 1:] |= grown[:, :-1]; g2[:, :-1] |= grown[:, 1:]
        found = pr is not None
        pos = math.hypot(pr[0] - po[0], pr[1] - po[1]) if found else math.inf
        if found and pr[2] is not None:
            head = abs(math.degrees((pr[2] - po[2] + math.pi) % (2 * math.pi) - math.pi))
        else:
            head = 180.0  # missing nose = heading failure
        st = es.track.route("outer").project(s.x, s.y).distance_m
        st = min(st, es.track.route("inner").project(s.x, s.y).distance_m)
        rows.append(dict(found=found, pos=pos, head=head,
                         psnr=psnr(orig, rec), psnr_car=psnr(orig, rec, g2), psnr_bg=psnr(orig, rec, ~g2),
                         section=es.arrays["branch"][e, t] != -1, fast=s.v >= 8.0, kerb=st >= 0.5))
    return summarise(rows, label)


def summarise(rows, label):
    def stats(rs):
        pos = np.array([r["pos"] for r in rs]); head = np.array([r["head"] for r in rs])
        found = np.mean([r["found"] for r in rs])
        fin = pos[np.isfinite(pos)]
        s = dict(n=len(rs), found=found,
                 pos_p95=np.percentile(fin, 95) if len(fin) else math.inf, pos_max=pos.max(),
                 head_p95=np.percentile(head, 95), head_max=head.max(),
                 psnr=np.mean([r["psnr"] for r in rs]), psnr_car=np.mean([r["psnr_car"] for r in rs]),
                 psnr_bg=np.mean([r["psnr_bg"] for r in rs]))
        s["pass"] = (found == 1.0 and s["pos_p95"] <= THRESH["pos_p95"] and s["pos_max"] <= THRESH["pos_max"]
                     and s["head_p95"] <= THRESH["head_p95"] and s["head_max"] <= THRESH["head_max"])
        return s

    out = {"all": stats(rows)}
    for k in ("section", "fast", "kerb"):
        sub = [r for r in rows if r[k]]
        if sub:
            out[k] = stats(sub)
    print(f"\n{label}")
    print(f"  {'slice':8s} {'n':>5s} {'found':>6s} {'pos p95':>8s} {'pos max':>8s} {'head p95':>9s} {'head max':>9s} "
          f"{'PSNR':>6s} {'car':>6s} {'bg':>6s}  verdict")
    for k, s in out.items():
        print(f"  {k:8s} {s['n']:5d} {100 * s['found']:5.1f}% {s['pos_p95']:8.3f} {s['pos_max']:8.3f} {s['head_p95']:9.1f} "
              f"{s['head_max']:9.1f} {s['psnr']:6.1f} {s['psnr_car']:6.1f} {s['psnr_bg']:6.1f}  {'PASS' if s['pass'] else 'FAIL'}")
    return out


# -- fake reconstructions -------------------------------------------------------------------
def copy(orig, s, track, rp, bg):
    return orig.copy()


def shift_1px(orig, s, track, rp, bg):
    return render(State(s.x + 1 / PX, s.y, s.theta, s.v), track, rp)


def erase(orig, s, track, rp, bg):
    return bg.copy()


def backwards(orig, s, track, rp, bg):
    return render(State(s.x, s.y, s.theta + math.pi, s.v), track, rp)


_noise_rng = np.random.default_rng(0)


def bg_noise(orig, s, track, rp, bg):
    return np.clip(orig.astype(int) + _noise_rng.integers(-4, 5, orig.shape), 0, 255).astype(np.uint8)


def block_avg(orig, s, track, rp, bg):
    b = orig.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3))
    return np.round(np.repeat(np.repeat(b, 8, 0), 8, 1)).astype(np.uint8)


def main():
    es = open_dataset("data/step3-v1", check=False).split("val")
    frames = eval_frames(es)
    track = es.track
    # baseline: estimator on originals vs the true State
    pe, he = [], []
    for e, t in frames:
        rp = es.render_params(e)
        s = State(*map(float, es.arrays["states"][e, t]))
        po = estimate(render(s, track, rp), render(State(-100, -100, 0, 0), track, rp), rp)
        pe.append(math.hypot(po[0] - s.x, po[1] - s.y))
        he.append(abs(math.degrees((po[2] - s.theta + math.pi) % (2 * math.pi) - math.pi)))
    pe, he = np.array(pe), np.array(he)
    print(f"Estimator on {len(frames)} original val Frames vs true State: "
          f"position median {np.median(pe):.3f} / p95 {np.percentile(pe, 95):.3f} / max {pe.max():.3f} m; "
          f"heading median {np.median(he):.1f} / p95 {np.percentile(he, 95):.1f} / max {he.max():.1f}°")
    for fn, label in [(copy, "exact copy (should PASS)"), (shift_1px, "car shifted 1 px = 0.625 m (should FAIL on position)"),
                      (erase, "car erased (should FAIL on found)"), (backwards, "car facing backwards (should FAIL on heading)"),
                      (bg_noise, "background noise ±4 per channel, car intact (should PASS)"),
                      (block_avg, "8×8 block average, enlarged (should FAIL)")]:
        evaluate(es, frames, fn, label)


if __name__ == "__main__":
    main()
