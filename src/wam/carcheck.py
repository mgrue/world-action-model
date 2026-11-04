"""The car-visibility check (Step 4 SPEC §4): does a Frame keep the car's position and heading?

Compares a reconstruction (or any predicted Frame) with its original Frame. It knows nothing
about the compressor, so later Steps can score predicted or Probe Frames with it. numpy only;
`contact_sheet` imports matplotlib lazily.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from toycar import RenderParams, State, Track, render

PX = 64 / 40.0  # pixels per metre
SEARCH_M = 4.0  # search square half-width around the original's estimate
CUTOFF = 9.0  # summed |ΔRGB| below this is ignored
FULL = 60.0  # summed |ΔRGB| at which a pixel counts fully as car
N_FRAMES = 4096
THRESH = {"pos_p95": 0.3, "pos_max": 0.6, "head_p95": 10.0, "head_max": 30.0}
READABLE_DEG = 20.0  # a Frame counts only if the estimator reads the ORIGINAL within this of the true heading
MAX_UNREADABLE = 0.01  # more unreadable Frames than this share → the check itself is INVALID
SLICES = ("all", "section", "fast", "kerb")
FAST_MPS = 8.0
KERB_M = 0.5

EMPTY = State(-100.0, -100.0, 0.0, 0.0)  # renders the empty Track

Pose = tuple[float, float, float | None]


def background(track: Track, rp: RenderParams) -> np.ndarray:
    """The empty-track render in the episode's colours."""
    return render(EMPTY, track, rp)


def car_region(frame: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Pixels where `frame` differs from `bg`, grown by 1 pixel in all 8 directions; (…, 64, 64) bool."""
    return grow((frame != bg).any(-1))


def grow(mask: np.ndarray) -> np.ndarray:
    """Grow a (…, 64, 64) bool mask by 1 pixel in every direction (incl. diagonals)."""
    g = mask.copy()
    g[..., 1:, :] |= mask[..., :-1, :]
    g[..., :-1, :] |= mask[..., 1:, :]
    h = g.copy()
    h[..., :, 1:] |= g[..., :, :-1]
    h[..., :, :-1] |= g[..., :, 1:]
    return h


def _wrap_deg(a: float) -> float:
    return abs(math.degrees((a + math.pi) % (2 * math.pi) - math.pi))


_YS, _XS = np.mgrid[0:64, 0:64] + 0.5


def estimate(frame: np.ndarray, background: np.ndarray, rp: RenderParams,
             centre: tuple[float, float] | None = None) -> Pose | None:
    """(x, y, θ or None) in metres / radians, or None if no car. `centre` (metres) limits the search."""
    f = frame.astype(np.float64)
    diff = np.abs(f - background.astype(np.float64)).sum(-1)
    w = np.clip(diff / FULL, 0, 1)
    w[diff < CUTOFF] = 0
    if centre is not None:
        r = SEARCH_M * PX
        w[(np.abs(_XS - centre[0] * PX) > r) | (np.abs(_YS - centre[1] * PX) > r)] = 0
    total = w.sum()
    if total < 1.0:
        return None
    db = np.linalg.norm(f - np.array(rp.body, float), axis=-1)
    dn = np.linalg.norm(f - np.array(rp.nose, float), axis=-1)
    wn = w * (dn < db)
    cx, cy = (w * _XS).sum() / total, (w * _YS).sum() / total
    if wn.sum() < 0.2:
        return cx / PX, cy / PX, None
    nx, ny = (wn * _XS).sum() / wn.sum(), (wn * _YS).sum() / wn.sum()
    return cx / PX, cy / PX, math.atan2(ny - cy, nx - cx)


def readable(po: Pose | None, theta: float) -> bool:
    """True if the estimate on the ORIGINAL Frame is within READABLE_DEG of the true heading."""
    return po is not None and po[2] is not None and _wrap_deg(po[2] - theta) <= READABLE_DEG


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        d = d[mask]
    mse = d.mean() if d.size else 0.0
    return 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)


def eval_frames(es, n: int = N_FRAMES) -> list[tuple[int, int]]:
    """Fixed (episode, tick) sample: per-episode quotas, ticks drawn with stream [seed, 5]."""
    E = len(es)
    out: list[tuple[int, int]] = []
    for i in range(E):
        q = n // E + (1 if i < n % E else 0)
        if q == 0:
            continue
        rng = np.random.default_rng([int(es.arrays["seed"][i]), 5])
        ticks = np.sort(rng.choice(int(es.arrays["length"][i]) + 1, size=q, replace=False))
        out += [(i, int(t)) for t in ticks]
    return out


# -- the check ------------------------------------------------------------------------------
def _as_state(s) -> State:
    return s if isinstance(s, State) else State(*(float(v) for v in s))


def score_frame(orig: np.ndarray, rec: np.ndarray, state: State, rp: RenderParams, bg: np.ndarray,
                track: Track, branch: int) -> dict:
    """One per-Frame row of §4.3."""
    po = estimate(orig, bg, rp)
    pr = estimate(rec, bg, rp, centre=None if po is None else po[:2])
    region = car_region(orig, bg)
    found = pr is not None and po is not None
    pos = math.hypot(pr[0] - po[0], pr[1] - po[1]) if found else math.inf
    head = _wrap_deg(pr[2] - po[2]) if found and pr[2] is not None and po[2] is not None else 180.0
    off = min(track.route("outer").project(state.x, state.y).distance_m,
              track.route("inner").project(state.x, state.y).distance_m)
    return {
        "found": found, "pos": pos, "head": head,
        "pos_true": math.hypot(pr[0] - state.x, pr[1] - state.y) if pr is not None else math.inf,
        "head_true": _wrap_deg(pr[2] - state.theta) if pr is not None and pr[2] is not None else 180.0,
        "psnr": psnr(orig, rec), "psnr_car": psnr(orig, rec, region), "psnr_bg": psnr(orig, rec, ~region),
        "section": bool(branch != -1), "fast": state.v >= FAST_MPS, "kerb": off >= KERB_M,
        "readable": readable(po, state.theta), "po": po, "pr": pr,
    }


def slice_stats(rows: Sequence[Mapping]) -> dict:
    """Stats over the readable rows of one slice; `skip` = unreadable rows left out."""
    rd = [r for r in rows if r.get("readable", True)]
    s: dict = {"n": len(rows), "skip": len(rows) - len(rd)}
    if not rd:
        s.update(found=0.0, pos_p95=math.inf, pos_max=math.inf, head_p95=180.0, head_max=180.0,
                 psnr=math.nan, psnr_car=math.nan, psnr_bg=math.nan, passed=False)
        return s
    pos = np.array([r["pos"] for r in rd], float)
    head = np.array([r["head"] for r in rd], float)
    fin = pos[np.isfinite(pos)]
    s.update(
        found=float(np.mean([r["found"] for r in rd])),
        pos_p95=float(np.percentile(fin, 95)) if len(fin) else math.inf, pos_max=float(pos.max()),
        head_p95=float(np.percentile(head, 95)), head_max=float(head.max()),
        psnr=float(np.mean([r["psnr"] for r in rd])), psnr_car=float(np.mean([r["psnr_car"] for r in rd])),
        psnr_bg=float(np.mean([r["psnr_bg"] for r in rd])),
    )
    s["passed"] = (s["found"] == 1.0 and s["pos_p95"] <= THRESH["pos_p95"] and s["pos_max"] <= THRESH["pos_max"]
                   and s["head_p95"] <= THRESH["head_p95"] and s["head_max"] <= THRESH["head_max"])
    return s


def summarise(rows: Sequence[Mapping], label: str | None = None) -> dict:
    """Per-slice stats with verdicts (§4.4); only `all` gates. Prints the table when `label` is given."""
    out: dict = {"all": slice_stats(rows)}
    a = out["all"]
    a["valid"] = a["skip"] <= MAX_UNREADABLE * a["n"]
    a["verdict"] = "INVALID" if not a["valid"] else ("PASS" if a["passed"] else "FAIL")
    for k in SLICES[1:]:
        sub = [r for r in rows if r[k]]
        if sub:
            out[k] = slice_stats(sub)
            out[k]["verdict"] = "PASS" if out[k]["passed"] else "FAIL"
    fin = [r["pos_true"] for r in rows if r.get("readable", True) and math.isfinite(r["pos_true"])]
    a["vs_true_pos_p95"] = float(np.percentile(fin, 95)) if fin else math.inf
    ht = [r["head_true"] for r in rows if r.get("readable", True)]
    a["vs_true_head_p95"] = float(np.percentile(ht, 95)) if ht else 180.0
    if label is not None:
        print_table(out, label)
    return out


def print_table(summary: Mapping, label: str) -> None:
    print(f"\n{label}")
    print(f"  {'slice':8s} {'n':>5s} {'skip':>4s} {'found':>6s} {'pos p95':>8s} {'pos max':>8s} {'head p95':>9s} "
          f"{'head max':>9s} {'PSNR':>6s} {'car':>6s} {'bg':>6s}  verdict")
    for k in SLICES:
        if k not in summary:
            continue
        s = summary[k]
        print(f"  {k:8s} {s['n']:5d} {s['skip']:4d} {100 * s['found']:5.1f}% {s['pos_p95']:8.3f} {s['pos_max']:8.3f} "
              f"{s['head_p95']:9.1f} {s['head_max']:9.1f} {s['psnr']:6.1f} {s['psnr_car']:6.1f} {s['psnr_bg']:6.1f}  "
              f"{s['verdict']}", flush=True)


def check(originals: np.ndarray, reconstructions: np.ndarray, states, render_params: Sequence[RenderParams],
          track: Track, *, branch, label: str | None = None) -> dict:
    """The full check over Frames aligned by index. Returns {slice: stats, …, "rows": per-Frame rows}."""
    n = len(originals)
    if not (len(reconstructions) == len(states) == len(render_params) == len(branch) == n):
        raise ValueError("originals, reconstructions, states, render_params and branch must align per Frame")
    bgs: dict[RenderParams, np.ndarray] = {}
    rows = []
    for i in range(n):
        rp = render_params[i]
        if rp not in bgs:
            bgs[rp] = background(track, rp)
        rows.append(score_frame(originals[i], reconstructions[i], _as_state(states[i]), rp, bgs[rp], track,
                                int(branch[i])))
    out = summarise(rows, label)
    out["rows"] = rows
    return out


def jsonable(summary: Mapping) -> dict:
    """The summary without per-Frame rows, with non-finite floats as None."""
    def fix(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (float, np.floating)):
            return float(v) if math.isfinite(v) else None
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    return {k: {kk: fix(vv) for kk, vv in v.items()} for k, v in summary.items() if k != "rows"}


# -- contact sheet (§4.5) -------------------------------------------------------------------
def contact_sheet(path: str | Path, title: str, clean: tuple, probes: Mapping[str, tuple]) -> None:
    """`clean` and each Probe entry are (rows, originals, reconstructions). Needs matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, orig, rec = clean
    rng = np.random.default_rng(0)
    n = len(rows)
    pick = [("random", int(i)) for i in rng.choice(n, min(8, n), replace=False)]
    pick += [("worst pos", int(i)) for i in np.argsort([-r["pos"] if r["readable"] else 0.0 for r in rows],
                                                       kind="stable")[:4]]
    pick += [("worst head", int(i)) for i in np.argsort([-r["head"] if r["readable"] else 0.0 for r in rows],
                                                        kind="stable")[:4]]
    for k in SLICES[1:]:
        idx = [i for i, r in enumerate(rows) if r[k]]
        pick += [(k, int(i)) for i in rng.choice(idx, min(2, len(idx)), replace=False)] if idx else []
    entries = [(lab, rows[i], orig[i], rec[i]) for lab, i in pick]
    for name, (prow, porig, prec) in probes.items():
        entries += [(name, prow[int(i)], porig[int(i)], prec[int(i)])
                    for i in rng.choice(len(prow), min(2, len(prow)), replace=False)]

    per_row = 5
    nr = max(1, math.ceil(len(entries) / per_row))
    fig, axes = plt.subplots(nr, per_row * 4, figsize=(per_row * 4 * 0.9, nr * 1.25 + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    h = 6
    for j, (lab, r, o, c) in enumerate(entries):
        a = axes[j // per_row, (j % per_row) * 4 : (j % per_row) * 4 + 4]
        ref = r["po"] or r["pr"] or (20.0, 20.0, None)
        cx, cy = ref[0] * PX, ref[1] * PX
        x0 = int(min(max(round(cx) - h, 0), 64 - 2 * h))
        y0 = int(min(max(round(cy) - h, 0), 64 - 2 * h))
        for k, img in enumerate([o, c, o[y0 : y0 + 2 * h, x0 : x0 + 2 * h], c[y0 : y0 + 2 * h, x0 : x0 + 2 * h]]):
            a[k].imshow(img, interpolation="nearest")
        for k, p in ((2, r["po"]), (3, r["pr"])):
            if p is not None and p[2] is not None:
                px, py = p[0] * PX - x0 - 0.5, p[1] * PX - y0 - 0.5
                a[k].arrow(px, py, 3 * math.cos(p[2]), 3 * math.sin(p[2]), color="cyan", width=0.25, head_width=1.2)
        a[0].set_title(lab, fontsize=6, loc="left")
        bad = r["pos"] > THRESH["pos_max"] or r["head"] > THRESH["head_max"]
        a[1].set_title(f"Δp {r['pos']:.2f} m  Δθ {r['head']:.0f}°", fontsize=6, loc="left",
                       color="red" if bad else "black")
    fig.suptitle(title + "   (each entry: original | reconstruction | zoom original | zoom reconstruction)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
