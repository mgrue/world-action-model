"""PROTOTYPE — throwaway. Steps 5–8 ticket 07: the Fork commit statistic, validated on labels and fakes.

References: train label Chunks of Windows approaching the fork section, direction-normalised steer
(steer × (+1 cw, −1 ccw)), grouped by entry offset e = first k (1..16) with branch[t+k] != -1, mean per
(e, Planned branch). A Chunk is classified by the nearer reference (L2 over the 16 steer values) and
committed if r = d_near / (d_near + d_far) < R. Fakes: labels (must classify correctly and commit), the
average of the two references (must not commit), shuffled labels.
"""
import json
import numpy as np
from toycar.dataset import open_dataset, INSTRUCTIONS

ds = open_dataset("data/step3-v1", check=False)
H = 16

def fork_windows(split):
    es = ds.split(split); w = es.windows(); a = es.arrays
    m = w.approaches_section
    e, t = w.index[m, 0].astype(np.int64), w.index[m, 1].astype(np.int64)
    k = np.arange(1, H + 1)
    br = a["branch"][e[:, None], t[:, None] + k]            # branch at ticks t+1..t+H (−1 outside section)
    entry = (br != -1).argmax(1) + 1                          # 1..16
    sgn = np.where(a["direction"][e] == 0, 1.0, -1.0)
    steer = a["clean"][e[:, None], t[:, None] + np.arange(H), 0] * sgn[:, None]
    return dict(e=e, t=t, entry=entry, pb=a["planned_branch"][e, t], instr=a["instruction"][e], steer=steer)

tr, va = fork_windows("train"), fork_windows("val")
ref = np.zeros((H + 1, 2, H))
for en in range(1, H + 1):
    for b in (0, 1):
        m = (tr["entry"] == en) & (tr["pb"] == b)
        ref[en, b] = tr["steer"][m].mean(0)

def classify(chunks, entry):
    r0, r1 = ref[entry, 0], ref[entry, 1]
    d0 = np.sqrt(((chunks - r0) ** 2).mean(-1)); d1 = np.sqrt(((chunks - r1) ** 2).mean(-1))
    cls = (d1 < d0).astype(int)
    r = np.minimum(d0, d1) / (d0 + d1 + 1e-9)
    return cls, r, np.sqrt(((r0 - r1) ** 2).mean(-1))

out = {"val_fork_windows": int(len(va["e"])), "by_entry": {}}
cls, r, sep = classify(va["steer"], va["entry"])
avg_cls, avg_r, _ = classify(0.5 * (ref[va["entry"], 0] + ref[va["entry"], 1]), va["entry"])
rng = np.random.default_rng(0)
sh = va["steer"][rng.permutation(len(va["e"]))]
sh_cls, sh_r, _ = classify(sh, va["entry"])
correct = cls == va["pb"]
for en in range(1, H + 1):
    m = va["entry"] == en
    out["by_entry"][en] = {"n": int(m.sum()), "ref_sep_rms": round(float(sep[m][0]), 3) if m.any() else None,
                           "label_acc": round(float(correct[m].mean()), 3) if m.any() else None,
                           "label_r_p95": round(float(np.quantile(r[m], 0.95)), 3) if m.any() else None}
for R in (0.25, 0.3, 0.35, 0.4):
    out[f"R{R}"] = {"labels_commit": round(float((r < R).mean()), 3),
                    "labels_commit_and_correct": round(float(((r < R) & correct).mean()), 3),
                    "average_commit": round(float((avg_r < R).mean()), 3),
                    "shuffled_commit": round(float((sh_r < R).mean()), 3)}
# restrict to Windows where the Branches have diverged enough to tell: entry ≤ 12 (≥ 4 Ticks inside the section)
for emax in (8, 10, 12, 16):
    m = va["entry"] <= emax
    out[f"entry<={emax}"] = {"n": int(m.sum()), "n_any": int((m & (va["instr"] == 2)).sum()),
                             "label_acc": round(float(correct[m].mean()), 4),
                             "labels_commit_R0.3": round(float((r[m] < 0.3).mean()), 4),
                             "average_commit_R0.3": round(float((avg_r[m] < 0.3).mean()), 4),
                             "shuffled_acc": round(float((sh_cls[m] == va["pb"][m]).mean()), 3)}
print(json.dumps(out, indent=1))
json.dump(out, open(".scratch/step5-8-model/prototypes/out/fork_metric.json", "w"), indent=1)
