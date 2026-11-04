"""PROTOTYPE — throwaway. Ticket 07: baseline action errors on the general val set
(eval_subset(val, 1024) minus Windows approaching the fork section), raw units, per dim, overall and k ≤ 4."""
import json
import numpy as np
from toycar.dataset import open_dataset

ds = open_dataset("data/step3-v1", check=False)
es = ds.split("val"); a = es.arrays
w = es.eval_subset(1024)
keep = ~w.approaches_section
e, t = w.index[keep, 0].astype(np.int64), w.index[keep, 1].astype(np.int64)
k = np.arange(16)
lab = a["clean"][e[:, None], t[:, None] + k]
exe = a["executed"][e[:, None], t[:, None] + k]
rng = np.random.default_rng(0)
fakes = {"zero": np.zeros_like(lab), "shuffled": lab[rng.permutation(len(lab))],
         "train_mean": np.broadcast_to(np.array([0.0, 0.077]), lab.shape),
         "executed": exe, "labels": lab}
for s in (0.05, 0.1, 0.2):
    fakes[f"labels+N(0,{s})"] = np.clip(lab + rng.normal(0, s, lab.shape), -1, 1)
def rmse(x, ks=slice(None)):
    return [round(float(v), 4) for v in np.sqrt(((x[:, ks] - lab[:, ks]) ** 2).mean((0, 1)))]
out = {"windows": int(keep.sum()), "of": int(len(w))}
for n, f in fakes.items():
    out[n] = {"rmse_steer_throttle": rmse(f), "rmse_k1-4": rmse(f, slice(0, 4))}
    print(f"{n:18s}", out[n])
json.dump(out, open(".scratch/step5-8-model/prototypes/out/action_baselines.json", "w"), indent=1)
