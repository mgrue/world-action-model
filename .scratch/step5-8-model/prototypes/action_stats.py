"""Ticket 04: action-label statistics on train Windows (H=16, clean-future filter). Arrays only, no rendering."""
import json, sys
import numpy as np
from toycar.dataset import open_dataset, INSTRUCTIONS

ds = open_dataset("data/step3-v1", check=False)
out = {}
for split in ("train", "val"):
    es = ds.split(split)
    w = es.windows()
    a = es.arrays
    e, t = w.index[:, 0].astype(np.int64), w.index[:, 1].astype(np.int64)
    k = np.arange(16)
    A = a["clean"][e[:, None], t[:, None] + k]           # (N, 16, 2)
    flat = A.reshape(-1, 2)
    s = {"windows": int(len(w))}
    for d, name in enumerate(("steer", "throttle")):
        x = flat[:, d]
        s[name] = {
            "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "q01_q50_q99": [float(v) for v in np.quantile(x, [0.01, 0.5, 0.99])],
            "frac_at_+1": float((x >= 0.999).mean()), "frac_at_-1": float((x <= -0.999).mean()),
            "hist_20bins_-1_1": np.histogram(x, bins=20, range=(-1, 1))[0].tolist(),
        }
    # chunk-level smoothness: std of step-to-step change
    dA = np.diff(A, axis=1)
    s["delta_std_per_tick"] = [float(v) for v in dA.reshape(-1, 2).std(0)]
    # the Fork, Instruction any: Windows that approach the fork section, split by the Branch the expert then took
    instr = a["instruction"][e]
    appr = w.approaches_section
    pb = a["planned_branch"][e, t]                  # planned branch at the Window's tick
    fork = {}
    for ii, iname in enumerate(INSTRUCTIONS):
        m = appr & (instr == ii)
        if not m.any():
            continue
        steer_mean = A[m, :, 0].mean(1)                 # mean steer over the Chunk
        groups = {}
        for b in np.unique(pb[m]):
            mb = m & (pb == b)
            groups[int(b)] = {"n": int(mb.sum()), "chunk_mean": A[mb].mean(0)[:, 0].round(3).tolist(),
                              "steer_mean_over_chunk": float(A[mb, :, 0].mean())}
        # separation of the two branch-conditional mean chunks vs within-group spread (steer)
        bs = sorted(groups)
        sep = None
        if len(bs) == 2:
            m0, m1 = (m & (pb == bs[0])), (m & (pb == bs[1]))
            mu0, mu1 = A[m0, :, 0].mean(0), A[m1, :, 0].mean(0)
            within = np.sqrt(0.5 * (A[m0, :, 0].var(0) + A[m1, :, 0].var(0)))
            sep = {"max_abs_mean_diff_steer": float(np.abs(mu0 - mu1).max()),
                   "at_k": int(np.abs(mu0 - mu1).argmax()) + 1,
                   "within_std_at_k": float(within[np.abs(mu0 - mu1).argmax()]),
                   "rms_dist_between_means_vs_avg": float(np.sqrt(((mu0 - mu1) ** 2).mean()))}
        fork[iname] = {"windows": int(m.sum()), "by_planned_branch": groups, "separation": sep,
                       "steer_mean_over_chunk_hist_-1_1": np.histogram(steer_mean, bins=20, range=(-1, 1))[0].tolist()}
    s["approaching_fork"] = fork
    out[split] = s
json.dump(out, open(".scratch/step5-8-model/prototypes/out/action_stats.json", "w"), indent=1)
tr = out["train"]
for n in ("steer", "throttle"):
    d = tr[n]; print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "hist_20bins_-1_1"})
    print("  hist", d["hist_20bins_-1_1"])
print("delta std per tick", tr["delta_std_per_tick"])
for iname, f in tr["approaching_fork"].items():
    print(iname, f["windows"], {b: (g["n"], round(g["steer_mean_over_chunk"], 3)) for b, g in f["by_planned_branch"].items()}, f["separation"])
    print("  chunk-mean-steer hist", f["steer_mean_over_chunk_hist_-1_1"])

# --- direction-normalised Fork analysis: steer × (+1 for direction 0, −1 for direction 1) ---
print("\n# direction-normalised steer (steer * sign), Windows approaching the fork section")
es = ds.split("train"); w = es.windows(); a = es.arrays
e, t = w.index[:, 0].astype(np.int64), w.index[:, 1].astype(np.int64)
A = a["clean"][e[:, None], t[:, None] + np.arange(16)]
sgn = np.where(a["direction"][e] == 0, 1.0, -1.0)
S = A[:, :, 0] * sgn[:, None]
instr = a["instruction"][e]; pb = a["planned_branch"][e, t]; appr = w.approaches_section
fork2 = {}
for ii, iname in enumerate(INSTRUCTIONS):
    m = appr & (instr == ii)
    r = {}
    for b in np.unique(pb[m]):
        mb = m & (pb == b)
        r[int(b)] = {"n": int(mb.sum()), "mean_by_k": S[mb].mean(0).round(3).tolist(), "std_by_k": S[mb].std(0).round(3).tolist()}
    if len(r) == 2:
        mu0, mu1 = (np.array(r[b]["mean_by_k"]) for b in sorted(r))
        sd = np.sqrt(0.5 * (np.array(r[0]["std_by_k"]) ** 2 + np.array(r[1]["std_by_k"]) ** 2))
        r["dprime_by_k"] = (np.abs(mu0 - mu1) / sd).round(2).tolist()
    cm = S[m].mean(1)
    r["chunk_mean_hist_-1_1"] = np.histogram(cm, bins=20, range=(-1, 1))[0].tolist()
    fork2[iname] = r
    print(iname, {b: v["n"] for b, v in r.items() if isinstance(b, int)})
    for b, v in r.items():
        if isinstance(b, int): print(f"  branch {b} mean_by_k", v["mean_by_k"])
    if "dprime_by_k" in r: print("  d' by k", r["dprime_by_k"])
    print("  chunk-mean hist", r["chunk_mean_hist_-1_1"])
out["train"]["approaching_fork_direction_normalised"] = fork2
json.dump(out, open(".scratch/step5-8-model/prototypes/out/action_stats.json", "w"), indent=1)
