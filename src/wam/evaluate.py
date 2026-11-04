"""Evaluation and checks for the world action model (SPEC §9).

Eval data is computed, not stored: `subsets(ds, horizon)` and `fork_references(ds)` are deterministic
functions of the dataset and fixed seeds, and their hash goes into `metrics.json` and the card.
All sampling uses the Policy it is given (EMA weights), 10 Euler steps, no CFG and fixed seeds.
matplotlib is imported lazily by `contact_sheet`.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toycar import State
from toycar.dataset import Windows

from . import carcheck as cc

# -- thresholds (SPEC §9.1–§9.3; set before any trained model, §9.6) -------------------------
STEP7 = {"vs_start": 0.5, "vs_min": 1.05}
GATES = {"rmse_steer": 0.15, "rmse_throttle": 0.15, "commit": 0.90, "follow": 0.90,
         "any_balance": (0.30, 0.70), "any_two_sided": 0.80}
COMMIT_R = 0.3
MAX_ENTRY = 12
REF_HORIZON = 16
INSTR_IDS = {"outer": 0, "inner": 1, "any": 2}
SIZES: dict = {
    "general": 1024,  # eval_subset(val, n) minus approaches_section
    "fork": {"any": 256, "outer": 128, "inner": 128},
    "samples": 16,
    "imagine": 256,
    "periodic_general": 192,
    "periodic_fork": {"any": 32, "outer": 16, "inner": 16},
    "sheet": 6,
}
SEEDS = {"general": 0, "fork": 1, "imagine": 2, "periodic_general": 3, "periodic_fork": 4, "sheet": 5}
SHEET_K = (4, 8, 12, 16)
CHUNK_ROWS = 256  # sampled rows (Windows × samples) per Policy call
CFG_SWEEP = (1.0, 1.5, 2.0, 3.0)
STEPS_SWEEP = (2, 4, 10, 20)


def sizes(eval_windows: int | None = None) -> dict:
    """The §9 set sizes; `eval_windows` (tests, smoke runs) caps every set."""
    if eval_windows is None:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in SIZES.items()}
    n = int(eval_windows)
    scale = lambda d: {k: max(2, round(v * n / SIZES["general"])) for k, v in d.items()}  # noqa: E731
    return {"general": n, "fork": scale(SIZES["fork"]), "samples": SIZES["samples"],
            "imagine": min(SIZES["imagine"], n), "periodic_general": min(SIZES["periodic_general"], n),
            "periodic_fork": scale(SIZES["periodic_fork"]), "sheet": SIZES["sheet"]}


# -- Fork windows and references (SPEC §9.3) ------------------------------------------------
@dataclass
class ForkInfo:
    """Per-Window Fork facts: entry offset e (0 = none), Planned branch, Instruction id, Direction sign,
    and the Direction-normalised label steer (n, H)."""

    entry: np.ndarray
    planned: np.ndarray
    instr: np.ndarray
    sign: np.ndarray
    steer: np.ndarray


def fork_info(w: Windows) -> ForkInfo:
    es, H = w.episodes, w.horizon
    a = es.arrays
    e, t = w.index[:, 0].astype(np.int64), w.index[:, 1].astype(np.int64)
    br = a["branch"][e[:, None], t[:, None] + np.arange(1, H + 1)]  # branch at t+1 … t+H
    inside = br != -1
    entry = np.where(inside.any(1), inside.argmax(1) + 1, 0)
    sign = np.where(a["direction"][e] == 0, 1.0, -1.0)
    steer = a["clean"][e[:, None], t[:, None] + np.arange(H), 0].astype(np.float64) * sign[:, None]
    return ForkInfo(entry.astype(np.int64), a["planned_branch"][e, t].astype(np.int64),
                    a["instruction"][e].astype(np.int64), sign, steer)


def fork_references(ds, horizon: int = REF_HORIZON) -> np.ndarray:
    """(17, 2, horizon) mean Direction-normalised train label steer per (entry offset e, Planned branch);
    row 0 and empty groups are NaN. Computed at H = 16 and truncated to `horizon` (SPEC §16.4)."""
    w = ds.split("train").windows(horizon=REF_HORIZON, future_frames=())
    w = w.subset(w.approaches_section)
    info = fork_info(w)
    ref = np.full((REF_HORIZON + 1, 2, REF_HORIZON), np.nan)
    for e in range(1, REF_HORIZON + 1):
        for b in (0, 1):
            m = (info.entry == e) & (info.planned == b)
            if m.any():
                ref[e, b] = info.steer[m].mean(0)
    return ref[:, :, :horizon]


def classify(steer: np.ndarray, r0: np.ndarray, r1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearer reference (0 outer, 1 inner) by RMS over the steer values, and r = d_near / (d_near + d_far)."""
    d0 = np.sqrt(((steer - r0) ** 2).mean(-1))
    d1 = np.sqrt(((steer - r1) ** 2).mean(-1))
    return (d1 < d0).astype(np.int64), np.minimum(d0, d1) / (d0 + d1 + 1e-9)


def fork_metrics(steer: np.ndarray, entry: np.ndarray, instr: np.ndarray, ref: np.ndarray) -> dict:
    """steer (n, S, H) Direction-normalised samples; entry, instr (n,). Commit, following, `any` balance and
    two-sided (SPEC §9.3); NaN where a set is empty."""
    n = len(entry)
    if n == 0:
        return {"commit": math.nan, "follow": math.nan, "any_balance": math.nan, "any_two_sided": math.nan,
                "n_windows": 0, "n_samples": 0}
    cls, r = classify(steer, ref[entry, 0][:, None], ref[entry, 1][:, None])
    oi, an = instr != INSTR_IDS["any"], instr == INSTR_IDS["any"]
    return {
        "commit": float((r < COMMIT_R).mean()),
        "follow": float((cls[oi] == instr[oi, None]).mean()) if oi.any() else math.nan,
        "any_balance": float((cls[an] == 0).mean()) if an.any() else math.nan,
        "any_two_sided": float((cls[an].min(1) != cls[an].max(1)).mean()) if an.any() else math.nan,
        "n_windows": int(n), "n_samples": int(steer.shape[1]),
    }


def general_metrics(pred: np.ndarray, labels: np.ndarray) -> dict:
    """pred (n, S, H, 2) raw, labels (n, H, 2): RMSE of sample 0 per dimension over all H and k ≤ 4,
    best-of-S and the per-k curves (SPEC §9.2, §9.5)."""
    if len(labels) == 0:
        return {k: math.nan for k in ("rmse_steer", "rmse_throttle", "rmse_steer_k4", "rmse_throttle_k4")} | {"n": 0}
    err = pred - labels[:, None]
    first = err[:, 0]
    rm = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    out = {"n": int(len(labels)), "rmse_steer": rm(first[..., 0]), "rmse_throttle": rm(first[..., 1]),
           "rmse_steer_k4": rm(first[:, :4, 0]), "rmse_throttle_k4": rm(first[:, :4, 1]),
           "per_k_steer": [rm(first[:, k, 0]) for k in range(err.shape[2])],
           "per_k_throttle": [rm(first[:, k, 1]) for k in range(err.shape[2])]}
    if err.shape[1] > 1:
        best = np.sqrt((err ** 2).mean((2, 3))).argmin(1)
        b = err[np.arange(len(err)), best]
        out.update(best_of=int(err.shape[1]), best_rmse_steer=rm(b[..., 0]), best_rmse_throttle=rm(b[..., 1]))
    return out


def baselines(labels: np.ndarray, mean: Sequence[float]) -> dict:
    """Reference predictions on the general set (report only, SPEC §9.2)."""
    rng = np.random.default_rng(0)
    rm = lambda p: {"rmse_steer": float(np.sqrt(np.mean((p - labels)[..., 0] ** 2))),  # noqa: E731
                    "rmse_throttle": float(np.sqrt(np.mean((p - labels)[..., 1] ** 2)))}
    if len(labels) == 0:
        return {}
    return {"train_mean": rm(np.broadcast_to(np.asarray(mean, float), labels.shape)),
            "zero": rm(np.zeros_like(labels)),
            "shuffled": rm(labels[rng.permutation(len(labels))]),
            "labels_noise_0.1": rm(labels + rng.normal(0, 0.1, labels.shape))}


# -- subsets (SPEC §9.2–§9.4, §16.4–§16.5) --------------------------------------------------
def _stratified(info: ForkInfo, ok: np.ndarray, counts: Mapping[str, int], tag: int, emax: int) -> np.ndarray:
    picks = []
    for name, n in counts.items():
        iid = INSTR_IDS[name]
        for j, e in enumerate(range(1, emax + 1)):
            cand = np.nonzero(ok & (info.instr == iid) & (info.entry == e))[0]
            q = min(n // emax + (1 if j < n % emax else 0), len(cand))
            if q:
                rng = np.random.default_rng([tag, iid, e])
                picks.append(np.sort(rng.choice(cand, size=q, replace=False)))
    return np.sort(np.concatenate(picks)) if picks else np.empty(0, np.int64)


@dataclass
class Subsets:
    general: Windows
    fork: Windows
    periodic_general: Windows
    periodic_fork: Windows
    hash: str


def subsets(ds, horizon: int = 16, sz: Mapping | None = None) -> Subsets:
    """The fixed val sets at horizon H: general, end-of-Run Fork, periodic general and periodic Fork."""
    sz = sz or sizes()
    es = ds.split("val")
    base = es.eval_subset(int(sz["general"]), horizon=horizon, future_frames=())
    general = base.subset(~base.approaches_section)
    fw = es.windows(horizon=horizon, future_frames=())
    fw = fw.subset(fw.approaches_section)
    info = fork_info(fw)
    emax = min(MAX_ENTRY, horizon)
    ok = (info.entry >= 1) & (info.entry <= emax)
    fork = fw.subset(_stratified(info, ok, sz["fork"], 7, emax))
    pfork = fw.subset(_stratified(info, ok, sz["periodic_fork"], 8, emax))
    perm = np.random.default_rng(9).permutation(len(general))
    pgen = general.subset(np.sort(perm[: int(sz["periodic_general"])]))
    h = hashlib.sha256()
    for name, w in (("general", general), ("fork", fork), ("periodic_general", pgen), ("periodic_fork", pfork)):
        h.update(name.encode())
        h.update(np.ascontiguousarray(w.index, np.int64).tobytes())
    h.update(str(horizon).encode())
    return Subsets(general, fork, pgen, pfork, h.hexdigest())


def eval_hash(sub: Subsets, ref: np.ndarray) -> str:
    h = hashlib.sha256(sub.hash.encode())
    h.update(np.ascontiguousarray(np.nan_to_num(ref, nan=-99.0), np.float64).tobytes())
    return h.hexdigest()


# -- Window inputs and sampling --------------------------------------------------------------
@dataclass
class Inputs:
    frames: np.ndarray  # (n, 64, 64, 3) uint8 observed Frames
    state: np.ndarray  # (n, 5) State features
    instr: np.ndarray  # (n,) Instruction ids
    labels: np.ndarray  # (n, H, 2) clean actions
    e: np.ndarray
    t: np.ndarray


def inputs(w: Windows) -> Inputs:
    """Observed Frame, State, Instruction and labels of every Window, without rendering future Frames."""
    es, H = w.episodes, w.horizon
    a = es.arrays
    e, t = w.index[:, 0].astype(np.int64), w.index[:, 1].astype(np.int64)
    n = len(e)
    frames = np.empty((n, 64, 64, 3), np.uint8)
    state = np.empty((n, 5), np.float32)
    for j in range(n):
        frames[j] = es._render(int(e[j]), int(t[j]), es.render_params(int(e[j])))
        state[j] = State(*map(float, a["states"][e[j], t[j]])).features()
    labels = a["clean"][e[:, None], t[:, None] + np.arange(H)].astype(np.float64)
    return Inputs(frames, state, a["instruction"][e].astype(np.int64), labels, e, t)


def sample_chunks(policy, x: Inputs, n_samples: int, seed: int, *, n_steps: int = 10, cfg_action: float = 1.0,
                  cfg_video: float = 1.0, decode: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """(actions (n, S, H, 2) raw, decoded imagined Frames (n, S, K, 64, 64, 3) | None), fixed seed."""
    import torch

    g = torch.Generator().manual_seed(seed)
    n = len(x.instr)
    per = max(1, CHUNK_ROWS // n_samples)
    acts, frames = [], []
    for i in range(0, n, per):
        sl = slice(i, i + per)
        res = policy.act(x.frames[sl], x.state[sl], [int(v) for v in x.instr[sl]], n_samples=n_samples,
                         n_steps=n_steps, cfg_action=cfg_action, cfg_video=cfg_video, seed=g, return_latents=decode)
        acts.append(res.actions)
        if decode and res.latents is not None:
            frames.append(policy.compressor.decode(res.latents).cpu().numpy())
    H = policy.horizon
    a = np.concatenate(acts) if acts else np.zeros((0, n_samples, H, 2), np.float32)
    f = np.concatenate(frames) if frames else None
    return a.astype(np.float64), f


def _fork_eval(policy, w: Windows, ref: np.ndarray, n_samples: int, seed: int, **kw) -> tuple[dict, np.ndarray]:
    x = inputs(w)
    info = fork_info(w)
    a, _ = sample_chunks(policy, x, n_samples, seed, **kw)
    steer = a[..., 0] * info.sign[:, None, None]
    return fork_metrics(steer, info.entry, info.instr, ref), steer


# -- imagined future (SPEC §9.5) -------------------------------------------------------------
def _pct(v: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(v, float), q)) if v else math.nan


def imagined_future(policy, w: Windows, seed: int) -> dict:
    """Decoded predicted Latents scored with `carcheck.score_frame` against the true future Frames, per k,
    next to the vae-v1 reconstruction floor and the true Frame itself."""
    if not policy.config.video_stream or len(w) == 0:
        return {}
    es, track = w.episodes, w.episodes.track
    a = es.arrays
    x = inputs(w)
    _, imag = sample_chunks(policy, x, 1, seed, decode=True)  # (n, 1, K, 64, 64, 3)
    ks = [int(k) for k in policy.future_k]
    true = np.empty((len(x.e), len(ks), 64, 64, 3), np.uint8)
    for j, (e, t) in enumerate(zip(x.e, x.t)):
        rp = es.render_params(int(e))
        for i, k in enumerate(ks):
            true[j, i] = es._render(int(e), int(t) + k, rp)
    rec = policy.compressor.reconstruct(true.reshape(-1, 64, 64, 3)).reshape(true.shape)
    bgs: dict = {}
    out: dict = {"n": int(len(x.e)), "per_k": {}}
    for i, k in enumerate(ks):
        rows: dict[str, list] = {"imagined": [], "reconstruction": [], "true": []}
        for j, (e, t) in enumerate(zip(x.e, x.t)):
            e, tk = int(e), int(t) + k
            rp = es.render_params(e)
            if e not in bgs:
                bgs[e] = cc.background(track, rp)
            st = State(*map(float, a["states"][e, tk]))
            br = int(a["branch"][e, tk])
            for name, fr in (("imagined", imag[j, 0, i]), ("reconstruction", rec[j, i]), ("true", true[j, i])):
                rows[name].append(cc.score_frame(true[j, i], fr, st, rp, bgs[e], track, br))
        out["per_k"][k] = {
            name: {"found": float(np.mean([r["found"] for r in rs])),
                   "pos_median": _pct([r["pos"] for r in rs], 50), "pos_p95": _pct([r["pos"] for r in rs], 95),
                   "head_median": _pct([r["head"] for r in rs], 50), "head_p95": _pct([r["head"] for r in rs], 95)}
            for name, rs in rows.items()}
    return out


# -- gates, Step 7, the full evaluation (SPEC §9) -------------------------------------------
def gates(general: Mapping, fork: Mapping) -> dict:
    vals = {"rmse_steer": general["rmse_steer"], "rmse_throttle": general["rmse_throttle"],
            "commit": fork["commit"], "follow": fork["follow"], "any_balance": fork["any_balance"],
            "any_two_sided": fork["any_two_sided"]}
    out = {}
    for k, v in vals.items():
        thr = GATES[k]
        v = float(v)
        if isinstance(thr, tuple):
            ok = math.isfinite(v) and thr[0] <= v <= thr[1]
        elif k.startswith("rmse"):
            ok = math.isfinite(v) and v <= thr
        else:
            ok = math.isfinite(v) and v >= thr
        out[k] = {"value": v, "threshold": list(thr) if isinstance(thr, tuple) else thr, "pass": bool(ok)}
    return out


def step7(rows: Sequence[Mapping], video: bool) -> dict:
    """Final EMA val loss per Stream ≤ 0.5 × its step-0 value and ≤ 1.05 × the lowest seen (SPEC §9.1)."""
    out: dict = {"thresholds": STEP7, "streams": {}}
    ok_all = True
    for s in (("video", "action") if video else ("action",)):
        col = f"ema_val_loss_{s}"
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        if not vals:
            out["streams"][s] = {"pass": False, "reason": "no val losses logged"}
            ok_all = False
            continue
        start, final, lo = vals[0], vals[-1], min(vals)
        ok = all(math.isfinite(v) for v in vals) and final <= STEP7["vs_start"] * start and final <= STEP7["vs_min"] * lo
        out["streams"][s] = {"start": start, "final": final, "min": lo, "ratio_start": final / start if start else None,
                             "pass": bool(ok)}
        ok_all &= ok
    out["verdict"] = "PASS" if ok_all else "FAIL"
    return out


def periodic(policy, sub: Subsets, ref: np.ndarray, sz: Mapping) -> dict:
    """The report-only periodic sampling metrics (SPEC §9.4)."""
    x = inputs(sub.periodic_general)
    a, _ = sample_chunks(policy, x, 1, SEEDS["periodic_general"])
    g = general_metrics(a, x.labels)
    f, _ = _fork_eval(policy, sub.periodic_fork, ref, int(sz["samples"]), SEEDS["periodic_fork"])
    return {"rmse_steer": g["rmse_steer"], "rmse_throttle": g["rmse_throttle"], "commit": f["commit"],
            "follow": f["follow"], "any_balance": f["any_balance"], "any_two_sided": f["any_two_sided"]}


def evaluate(policy, ds, *, sz: Mapping | None = None, sweeps: bool = False, quiet: bool = False,
             sub: Subsets | None = None, ref: np.ndarray | None = None) -> dict:
    """The §9.2–§9.5 gates and report (Step 7's check is added by the caller from the Run's log)."""
    t0 = time.time()
    sz = sz or sizes()
    H = policy.horizon
    ref = fork_references(ds, H) if ref is None else ref
    sub = subsets(ds, H, sz) if sub is None else sub
    S = int(sz["samples"])

    x = inputs(sub.general)
    a, _ = sample_chunks(policy, x, S, SEEDS["general"])
    general = general_metrics(a, x.labels)
    fork, _ = _fork_eval(policy, sub.fork, ref, S, SEEDS["fork"])
    g = gates(general, fork)
    report: dict = {"baselines": baselines(x.labels, policy.config.action_mean)}
    w_im = sub.general.subset(np.arange(min(int(sz["imagine"]), len(sub.general))))
    report["imagined_future"] = imagined_future(policy, w_im, SEEDS["imagine"])
    if sweeps:
        report["sweeps"] = run_sweeps(policy, sub, ref, S, x)
    verdict = "PASS" if all(v["pass"] for v in g.values()) else "FAIL"
    out = {"gates": g, "verdict": verdict, "general": general, "fork": fork, "report": report,
           "thresholds": {"gates": {k: (list(v) if isinstance(v, tuple) else v) for k, v in GATES.items()},
                          "commit_r": COMMIT_R, "max_entry": MAX_ENTRY, "step7": STEP7},
           "sizes": dict(sz), "eval_data_sha256": eval_hash(sub, ref), "eval_s": round(time.time() - t0, 1)}
    if not quiet:
        print_report(out)
    return out


def run_sweeps(policy, sub: Subsets, ref: np.ndarray, S: int, x: Inputs) -> dict:
    """CFG w × (both / action / video) → following and commit; n_steps → general RMSE and commit (SPEC §9.5)."""
    out: dict = {"cfg": {}, "n_steps": {}}
    if policy.config.objective == "regression":
        return out
    modes = ("both", "action", "video") if policy.config.video_stream else ("action",)
    for w in CFG_SWEEP:
        for mode in modes:
            ca = w if mode in ("both", "action") else 1.0
            cv = w if mode in ("both", "video") else 1.0
            f, _ = _fork_eval(policy, sub.fork, ref, S, SEEDS["fork"], cfg_action=ca, cfg_video=cv)
            out["cfg"][f"{mode}@{w}"] = {"follow": f["follow"], "commit": f["commit"]}
    for n in STEPS_SWEEP:
        a, _ = sample_chunks(policy, x, 1, SEEDS["general"], n_steps=n)
        g = general_metrics(a, x.labels)
        f, _ = _fork_eval(policy, sub.fork, ref, S, SEEDS["fork"], n_steps=n)
        out["n_steps"][n] = {"rmse_steer": g["rmse_steer"], "rmse_throttle": g["rmse_throttle"], "commit": f["commit"]}
    return out


def _fmt(v) -> str:
    if isinstance(v, list):
        return f"[{v[0]:.2f}, {v[1]:.2f}]"
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4f}"


def print_report(m: Mapping, step7_result: Mapping | None = None) -> None:
    if step7_result is not None:
        parts = [f"{s} {d.get('final', math.nan):.4f}/{d.get('start', math.nan):.4f}" if "final" in d else f"{s} —"
                 for s, d in step7_result["streams"].items()]
        print(f"\nStep 7 loss check (final / step-0 EMA val): {', '.join(parts)}  {step7_result['verdict']}")
    print(f"\n  {'gate':16s} {'value':>8s} {'threshold':>14s}  result")
    for k, g in m["gates"].items():
        thr = g["threshold"]
        op = "in" if isinstance(thr, list) else ("≤" if k.startswith("rmse") else "≥")
        print(f"  {k:16s} {_fmt(g['value']):>8s} {op:>2s} {_fmt(thr):>11s}  {'PASS' if g['pass'] else 'FAIL'}")
    gen = m["general"]
    print(f"\n  general set: {gen['n']} Windows; k ≤ 4 RMSE steer {_fmt(gen.get('rmse_steer_k4'))} "
          f"throttle {_fmt(gen.get('rmse_throttle_k4'))}"
          + (f"; best-of-{gen['best_of']} {_fmt(gen['best_rmse_steer'])} / {_fmt(gen['best_rmse_throttle'])}"
             if "best_of" in gen else ""))
    for name, b in m["report"].get("baselines", {}).items():
        print(f"    baseline {name:18s} steer {_fmt(b['rmse_steer'])}  throttle {_fmt(b['rmse_throttle'])}")
    print(f"  Fork set: {m['fork']['n_windows']} Windows × {m['fork']['n_samples']} samples")
    im = m["report"].get("imagined_future") or {}
    if im:
        print(f"  imagined future ({im['n']} Windows): pos median / p95 m, head median / p95 °, [vae floor]")
        for k, d in im["per_k"].items():
            i, r = d["imagined"], d["reconstruction"]
            print(f"    k={k:2d}  {_fmt(i['pos_median'])} / {_fmt(i['pos_p95'])}   {_fmt(i['head_median'])} / "
                  f"{_fmt(i['head_p95'])}   [{_fmt(r['pos_median'])} / {_fmt(r['pos_p95'])}]")
    if "sweeps" in m["report"]:
        for k, d in m["report"]["sweeps"]["cfg"].items():
            print(f"    cfg {k:12s} follow {_fmt(d['follow'])} commit {_fmt(d['commit'])}")
        for k, d in m["report"]["sweeps"]["n_steps"].items():
            print(f"    n_steps {k:3d}  rmse {_fmt(d['rmse_steer'])} / {_fmt(d['rmse_throttle'])} commit {_fmt(d['commit'])}")
    print(f"\n§9 gates: {m['verdict']}", flush=True)


def jsonable(obj):
    """Recursively replace non-finite floats with None and numpy scalars with Python ones."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


# -- contact sheet (SPEC §9.5) ---------------------------------------------------------------
def sheet_data(policy, ds, sub: Subsets, ref: np.ndarray, sz: Mapping | None = None) -> dict:
    sz = sz or sizes()
    n, S = int(sz["sheet"]), int(sz["samples"])
    gw = sub.general.subset(np.linspace(0, len(sub.general) - 1, min(n, len(sub.general))).astype(np.int64)) \
        if len(sub.general) else sub.general
    x = inputs(gw)
    a, imag = sample_chunks(policy, x, S, SEEDS["sheet"], decode=policy.config.video_stream)
    es = gw.episodes
    ks = [k for k in SHEET_K if k <= policy.horizon]
    true = np.stack([np.stack([es._render(int(e), int(t) + k, es.render_params(int(e))) for k in ks])
                     for e, t in zip(x.e, x.t)]) if len(x.e) else np.zeros((0, len(ks), 64, 64, 3), np.uint8)
    fk = [int(k) for k in policy.future_k]
    im = None
    if imag is not None:
        im = np.stack([imag[:, 0, fk.index(k)] if k in fk else np.zeros_like(imag[:, 0, 0]) for k in ks], 1)
    info_all = fork_info(sub.fork)
    anyi = np.nonzero(info_all.instr == INSTR_IDS["any"])[0]
    pick = anyi[np.linspace(0, len(anyi) - 1, min(n, len(anyi))).astype(np.int64)] if len(anyi) else anyi
    fw = sub.fork.subset(pick)
    _, steer = _fork_eval(policy, fw, ref, S, SEEDS["sheet"]) if len(fw) else ({}, np.zeros((0, S, policy.horizon)))
    finfo = fork_info(fw)
    return {"obs": x.frames, "true": true, "imagined": im, "ks": ks, "labels": x.labels, "samples": a,
            "fork_steer": steer, "fork_entry": finfo.entry, "fork_ref": ref}


def contact_sheet(path: str | Path, title: str, d: Mapping) -> None:
    """Three panels: general Windows' Frames (true above, imagined below), their Chunks vs labels, and the
    `any` Fork Windows' Direction-normalised steer samples over the two references. Needs matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, ks = len(d["obs"]), d["ks"]
    nf = len(d["fork_steer"])
    fig = plt.figure(figsize=(16, 4 + 2.2 * max(n, 1) + 3), constrained_layout=True)
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[1.4 * max(n, 1), 1.0 * max(n, 1), 3.0])
    top.suptitle(f"{title} — observed Frame, true future (row 1) and imagined (row 2) at k = {ks}")
    cols = 1 + len(ks)
    axs = top.subplots(2 * max(n, 1), cols, squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for i in range(n):
        axs[2 * i, 0].imshow(d["obs"][i])
        axs[2 * i, 0].set_title("obs", fontsize=7)
        for j, k in enumerate(ks):
            axs[2 * i, 1 + j].imshow(d["true"][i, j])
            axs[2 * i, 1 + j].set_title(f"true k={k}", fontsize=7)
            if d["imagined"] is not None:
                axs[2 * i + 1, 1 + j].imshow(d["imagined"][i, j])
                axs[2 * i + 1, 1 + j].set_title(f"imagined k={k}", fontsize=7)
    mid.suptitle("Chunks: label (black) vs samples (colour)")
    axm = mid.subplots(max(n, 1), 2, squeeze=False)
    for i in range(n):
        k = np.arange(1, d["labels"].shape[1] + 1)
        for c, name in enumerate(("steer", "throttle")):
            ax = axm[i, c]
            for s in range(d["samples"].shape[1]):
                ax.plot(k, d["samples"][i, s, :, c], color="tab:blue", alpha=0.3, lw=0.8)
            ax.plot(k, d["labels"][i, :, c], color="black", lw=1.6)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(name, fontsize=7)
            ax.tick_params(labelsize=6)
    bot.suptitle("`any` Fork Windows: Direction-normalised steer samples over the outer (red) / inner (green) references")
    axb = bot.subplots(1, max(nf, 1), squeeze=False)[0]
    if nf == 0:
        axb[0].axis("off")
        axb[0].text(0.5, 0.5, "no `any` Fork Windows in this set", ha="center", va="center")
    for i in range(nf):
        ax = axb[i]
        e = int(d["fork_entry"][i])
        k = np.arange(1, d["fork_steer"].shape[2] + 1)
        for s in range(d["fork_steer"].shape[1]):
            ax.plot(k, d["fork_steer"][i, s], color="tab:blue", alpha=0.35, lw=0.8)
        ax.plot(k, d["fork_ref"][e, 0], color="tab:red", lw=2)
        ax.plot(k, d["fork_ref"][e, 1], color="tab:green", lw=2)
        ax.set_title(f"entry e={e}", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=80)
    plt.close(fig)
