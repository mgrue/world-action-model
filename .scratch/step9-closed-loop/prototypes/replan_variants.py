"""Throwaway prototype for ticket 03 "Replanning scheme": closed-loop variants on a dev seed range.

Each variant runs 24 lockstep-batched episodes (4 per Instruction × Direction) on seeds
CLOSED_LOOP_SEED_BLOCK + 1000 + k (kept apart from 01's seeds 0…59 and from any later gate set).
Initial noise is drawn per episode row, so it doesn't depend on which episodes are still alive:
  seed_mode per_replan:  generator [episode seed, tick]  — fresh noise every replan
  seed_mode per_episode: generator [episode seed]        — identical noise at every replan
Fork flips are measured post hoc: for every replan 1…12 Ticks before entering a Branch, its executed Chunk's
Direction-normalised steer is classified against the Step 5–8 Fork references; a Passage approach "flips"
when its replans disagree.

    uv run python .scratch/step9-closed-loop/prototypes/replan_variants.py [variant ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from naive_loop import COMBOS, OUT, show, table  # noqa: E402

from toycar.dataset import CLOSED_LOOP_SEED_BLOCK, open_dataset, sample_render_params  # noqa: E402
from toycar.env import CarEnv  # noqa: E402
from toycar.expert import score_episode, start_spec  # noqa: E402
from wam import evaluate as E  # noqa: E402
from wam import flow  # noqa: E402
from wam.policy import INSTRUCTION_IDS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEV = [(CLOSED_LOOP_SEED_BLOCK + 1000 + k, *COMBOS[k % 6]) for k in range(24)]

VARIANTS = {
    "k1": dict(k=1),
    "k1_ep": dict(k=1, seed_mode="per_episode"),
    "k2": dict(k=2),
    "k2_ep": dict(k=2, seed_mode="per_episode"),
    "k2_coh8": dict(k=2, n=8, select="coherence"),
    "k1_ens": dict(k=1, ensemble=0.01),
    "k1_steps4": dict(k=1, n_steps=4),
    "k1_cfg2": dict(k=1, cfg_action=2.0),
    "k4": dict(k=4),
    "k4_coh8": dict(k=4, n=8, select="coherence"),
    "k1_coh4_s4": dict(k=1, n=4, select="coherence", n_steps=4),
    "k1_coh4_s4_cfg2": dict(k=1, n=4, select="coherence", n_steps=4, cfg_action=2.0),
}


def noise(cfg, seeds, ticks, mode):
    xs_z, xs_a = [], []
    for s, t in zip(seeds, ticks):
        g = torch.Generator().manual_seed(int(s) if mode == "per_episode" else int(s) * 100_003 + int(t))
        z, a = flow.initial_noise(cfg, 1, g)
        xs_z.append(z), xs_a.append(a)
    return (torch.cat(xs_z) if xs_z[0] is not None else None), torch.cat(xs_a)


@torch.no_grad()
def chunks(pol, frames, feats, instr, seeds, tick, *, n, n_steps, cfg_action, seed_mode):
    """(B, n, 16, 2) clipped actions; sample i of row j uses seed (seed_j, i)."""
    cfg, dev = pol.config, pol.device
    obs = pol.compressor.encode(torch.from_numpy(frames)).clone()
    rep = lambda x: x.repeat_interleave(n, 0)  # noqa: E731
    sd = [s * 64 + i for s in seeds for i in range(n)]
    x_z, x_a = noise(cfg, sd, [tick] * len(sd), seed_mode)
    st = torch.from_numpy(feats).to(dev)
    ins = torch.tensor([INSTRUCTION_IDS[i] for i in instr], device=dev)
    _, a = flow.euler(flow.guided(pol.model, cfg_action, 1.0), rep(obs.to(dev)), rep(st), rep(ins),
                      None if x_z is None else x_z.to(dev), x_a.to(dev), n_steps=n_steps, t_shift=cfg.t_shift)
    a = flow.denormalise(a, cfg).clamp(-1, 1)
    return a.reshape(len(seeds), n, cfg.horizon, 2).float().cpu().numpy()


def pick(cands, prev, k, select, rho=0.9):
    """cands (n, 16, 2); prev = previously selected Chunk or None."""
    if select == "first" or prev is None:
        return cands[0]
    ov = 16 - k
    w = rho ** np.arange(ov)
    d = (np.linalg.norm(cands[:, :ov] - prev[None, k:], axis=-1) * w).sum(-1)
    return cands[int(d.argmin())]


def run(pol, eps, *, k=1, n=1, select="first", seed_mode="per_replan", n_steps=10, cfg_action=1.0, ensemble=None):
    envs, states, infos, queue, prev, hist, recs = [], [], [], [], [], [], []
    for seed, ins, d in eps:
        env = CarEnv()
        s, i = env.reset(seed, start_spec(ins, d), render_params=sample_render_params(seed))
        envs.append(env), states.append(s), infos.append([i]), queue.append([]), prev.append(None), hist.append([])
        recs.append({"seed": seed, "instruction": ins, "direction": d, "lat": [], "replans": []})
    alive, tick, n_calls, t0 = list(range(len(eps))), 0, 0, time.perf_counter()
    while alive:
        need = [j for j in alive if not queue[j]]
        if need:
            c = chunks(pol, np.stack([envs[j].render() for j in need]), np.stack([states[j].features() for j in need]),
                       [eps[j][1] for j in need], [eps[j][0] for j in need], tick, n=n, n_steps=n_steps,
                       cfg_action=cfg_action, seed_mode=seed_mode)
            n_calls += 1
            for row, j in enumerate(need):
                ch = pick(c[row], prev[j], k, select)
                prev[j] = ch
                recs[j]["replans"].append((tick, ch[:, 0].tolist()))
                if ensemble is None:
                    queue[j] = [tuple(a) for a in ch[:k]]
                else:  # ACT temporal ensembling (k = 1): average every live Chunk's prediction for this Tick
                    hist[j].append((tick, ch))
                    hist[j] = [(t, h) for t, h in hist[j] if tick - t < 16]
                    preds = np.stack([h[tick - t] for t, h in hist[j]])  # oldest first
                    w = np.exp(-ensemble * np.arange(len(preds)))
                    queue[j] = [tuple((preds * w[:, None]).sum(0) / w.sum())]
        for j in list(alive):
            st = envs[j].step(queue[j].pop(0))
            states[j] = st.state
            infos[j].append(st.info)
            recs[j]["lat"].append(envs[j].track.distance_to_centerline(st.state.x, st.state.y))
            if st.terminated or st.truncated:
                alive.remove(j)
                recs[j]["score"] = score_episode(envs[j].track, eps[j][1], infos[j])
                recs[j]["branch"] = [-1 if i.branch is None else (0 if i.branch == "outer" else 1) for i in infos[j]]
                recs[j]["crash_in_section"] = bool(recs[j]["score"].crashed and infos[j][-1].branch is not None)
        tick += 1
    return recs, {"calls": n_calls, "wall_s": time.perf_counter() - t0}


def fork_flips(recs, ref):
    """Per Passage approach: classes of the replans 1…12 Ticks before entry; flip if they disagree."""
    out = {"approaches": 0, "flips": 0, "any_approaches": 0, "any_flips": 0, "uncommitted": 0, "classified": 0}
    for r in recs:
        br = np.array(r["branch"])
        sign = 1.0 if r["direction"] == "cw" else -1.0
        entries = [t for t in range(1, len(br)) if br[t] != -1 and br[t - 1] == -1]
        for te in entries:
            cls = []
            for t, steer in r["replans"]:
                e = te - t
                if 1 <= e <= E.MAX_ENTRY:
                    c, rr = E.classify(np.array(steer) * sign, ref[e, 0], ref[e, 1])
                    cls.append(int(c))
                    out["classified"] += 1
                    out["uncommitted"] += int(rr >= E.COMMIT_R)
            if len(cls) < 2:
                continue
            flip = len(set(cls)) > 1
            out["approaches"] += 1
            out["flips"] += flip
            if r["instruction"] == "any":
                out["any_approaches"] += 1
                out["any_flips"] += flip
    return out


def main(names):
    import wam.policy
    pol = wam.policy.load("wam-v1")
    ref = E.fork_references(open_dataset(REPO / "data/step3-v1", check=False))
    path = OUT / "replan_variants.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        recs, t = run(pol, DEV, **VARIANTS[name])
        rows = table(recs)
        ff = fork_flips(recs, ref)
        s = rows["all/all"]
        s["crash_in_section"] = sum(r["crash_in_section"] for r in recs)
        show({"all/all": s, "any/cw": rows["any/cw"], "any/ccw": rows["any/ccw"]},
             f"{name} {VARIANTS[name]}: wall {t['wall_s']:.0f} s, {t['calls']} calls")
        print(f"  fork approaches {ff['approaches']}, flips {ff['flips']} (any: {ff['any_flips']}/{ff['any_approaches']}),"
              f" uncommitted replans {ff['uncommitted']}/{ff['classified']}", flush=True)
        res[name] = {"params": VARIANTS[name], "all": s, "rows": rows, "fork": ff, **t,
                     "crash_ticks": [r["score"].steps for r in recs if r["score"].crashed]}
        path.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
