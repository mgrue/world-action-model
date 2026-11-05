"""Throwaway prototype for ticket 01 "Does wam-v1 drive at all?".

Naive closed loop: render Frame -> policy.act(..., n_samples=1) -> execute the first k actions -> replan.
Episodes run in lockstep and are batched into one `act` call per replan. The expert runs through the same loop
(replan every Tick, k=1) as a parity check against toycar.expert.run_episode.

    uv run python .scratch/step9-closed-loop/prototypes/naive_loop.py main      # 60 episodes, k=4, + expert parity
    uv run python .scratch/step9-closed-loop/prototypes/naive_loop.py sweep     # k in 1/4/8/16 on 24 episodes
    uv run python .scratch/step9-closed-loop/prototypes/naive_loop.py timing    # act() latency B=1 vs batched
    uv run --group viz python .scratch/step9-closed-loop/prototypes/naive_loop.py plot   # trajectories PNG
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

from toycar.dataset import CLOSED_LOOP_SEED_BLOCK, sample_render_params
from toycar.env import CarEnv
from toycar.expert import Expert, ExpertParams, run_episode, score_episode, start_spec, summarize

OUT = Path(__file__).parent / "out"
COMBOS = list(product(["outer", "inner", "any"], ["cw", "ccw"]))


def episodes(n_per_combo: int):
    """(seed, instruction, direction) — episode k of the closed-loop block, combos interleaved."""
    out = []
    for k in range(n_per_combo * len(COMBOS)):
        ins, d = COMBOS[k % len(COMBOS)]
        out.append((CLOSED_LOOP_SEED_BLOCK + k, ins, d))
    return out


def run_policy(policy, eps, k: int, seed_base: int = 0, max_steps: int = 900, keep_traj: bool = False):
    """Lockstep closed loop for all episodes; returns per-episode records."""
    envs, recs, states, infos, queues = [], [], [], [], []
    for seed, ins, d in eps:
        env = CarEnv(max_steps=max_steps)
        s, i = env.reset(seed, start_spec(ins, d), render_params=sample_render_params(seed))
        envs.append(env)
        states.append(s)
        infos.append([i])
        queues.append([])
        recs.append({"seed": seed, "instruction": ins, "direction": d, "traj": [(s.x, s.y)], "lat": [], "steer": []})
    alive = list(range(len(eps)))
    n_calls, t_act = 0, 0.0
    tick = 0
    while alive:
        need = [j for j in alive if not queues[j]]
        if need:
            frames = np.stack([envs[j].render() for j in need])
            feats = np.stack([states[j].features() for j in need])
            ins = [eps[j][1] for j in need]
            t0 = time.perf_counter()
            r = policy.act(frames, feats, ins, n_samples=1, seed=seed_base * 1_000_003 + tick)
            t_act += time.perf_counter() - t0
            n_calls += 1
            for row, j in enumerate(need):
                queues[j] = [tuple(a) for a in r.actions[row, 0, :k]]
        for j in list(alive):
            a = queues[j].pop(0)
            st = envs[j].step(a)
            states[j] = st.state
            infos[j].append(st.info)
            rec = recs[j]
            rec["steer"].append(float(a[0]))
            rec["lat"].append(envs[j].track.distance_to_centerline(st.state.x, st.state.y))
            if keep_traj:
                rec["traj"].append((st.state.x, st.state.y))
            if st.terminated or st.truncated:
                alive.remove(j)
                sc = score_episode(envs[j].track, eps[j][1], infos[j])
                rec["score"] = sc
                if sc.crashed:
                    last = st.info
                    rec["crash"] = {"tick": last.steps, "station_m": last.station_m, "branch": last.branch,
                                    "lat": rec["lat"][-1], "x": st.state.x, "y": st.state.y,
                                    "in_fork_section": envs[j].track.fork_station_m <= last.station_m
                                    < envs[j].track.merge_station_m}
        tick += 1
    return recs, {"calls": n_calls, "t_act_s": t_act}


def table(recs):
    rows = {}
    for ins, d in COMBOS + [("all", "all")]:
        sel = [r for r in recs if (ins == "all" or (r["instruction"] == ins and r["direction"] == d))]
        if not sel:
            continue
        s = summarize([r["score"] for r in sel])
        lat = np.concatenate([r["lat"] for r in sel])
        s["lat_mean_m"] = float(lat.mean())
        s["lat_p95_m"] = float(np.percentile(lat, 95))
        s["crash_in_fork_section"] = sum(1 for r in sel if r.get("crash", {}).get("in_fork_section"))
        rows[f"{ins}/{d}"] = s
    return rows


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'combo':10} {'eps':>4} {'crash':>6} {'laps':>5} {'minlap':>6} {'follow':>6} {'success':>7} "
          f"{'wrongP':>6} {'outer%':>6} {'lat':>5} {'lat95':>5} {'fkCr':>4}")
    for k, s in rows.items():
        of = "" if s["outer_fraction"] is None else f"{100 * s['outer_fraction']:.0f}"
        print(f"  {k:10} {s['episodes']:4d} {s['crash_rate']:6.2f} {s['laps_mean']:5.2f} {s['laps_min']:6d} "
              f"{s['instruction_followed_rate']:6.2f} {s['success_rate']:7.2f} {s['wrong_passages']:6d} {of:>6} "
              f"{s['lat_mean_m']:5.2f} {s['lat_p95_m']:5.2f} {s['crash_in_fork_section']:4d}")


def jsonable(recs):
    out = []
    for r in recs:
        d = {k: v for k, v in r.items() if k not in ("traj", "lat", "steer", "score")}
        d["score"] = asdict(r["score"])
        out.append(d)
    return out


def expert_loop(seed, ins, d):
    """The expert through the same loop structure (render each Tick, act, step)."""
    env = CarEnv()
    s, i = env.reset(seed, start_spec(ins, d), render_params=sample_render_params(seed))
    ex = Expert(env.track, ins, ExpertParams())
    ex.reset(seed, s, i)
    infos = [i]
    while True:
        env.render()  # same work as a Policy loop; must not change anything
        r = env.step(ex.act(s, i).executed)
        s, i = r.state, r.info
        infos.append(i)
        if r.terminated or r.truncated:
            return score_episode(env.track, ins, infos)


def cmd_main():
    import wam.policy
    pol = wam.policy.load("wam-v1")
    eps = episodes(10)
    parity = [expert_loop(*e) == run_episode(*e)[0] for e in eps]
    ex_rows = table([{"score": expert_loop(*e), "instruction": e[1], "direction": e[2], "lat": [0.0]} for e in eps])
    show(ex_rows, "expert (same seeds; lat columns not measured)")
    print(f"expert parity with run_episode: {sum(parity)}/{len(parity)}")
    t0 = time.perf_counter()
    recs, t = run_policy(pol, eps, k=4)
    wall = time.perf_counter() - t0
    rows = table(recs)
    show(rows, f"wam-v1, k=4, n_samples=1, 60 episodes; wall {wall:.0f} s, {t['calls']} act calls, {t['t_act_s']:.0f} s in act")
    for r in recs:
        if "crash" in r:
            c = r["crash"]
            print(f"  crash seed {r['seed']} {r['instruction']}/{r['direction']}: tick {c['tick']} station "
                  f"{c['station_m']:.1f} branch {c['branch']} fork-section {c['in_fork_section']} lat {c['lat']:.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "naive_main.json").write_text(json.dumps(
        {"expert": ex_rows, "expert_parity": [sum(parity), len(parity)], "wam_k4": rows, "episodes": jsonable(recs),
         "wall_s": wall, **t}, indent=1, default=str))


def cmd_sweep():
    import wam.policy
    pol = wam.policy.load("wam-v1")
    eps = episodes(4)
    res = {}
    for k in (1, 4, 8, 16):
        t0 = time.perf_counter()
        recs, t = run_policy(pol, eps, k=k, seed_base=1)
        rows = table(recs)
        show({"all/all": rows["all/all"], **{c: rows[c] for c in rows if c.startswith("any")}},
             f"k={k}: wall {time.perf_counter() - t0:.0f} s, {t['calls']} calls")
        res[k] = rows
    (OUT / "naive_sweep.json").write_text(json.dumps(res, indent=1, default=str))


def cmd_timing():
    import torch
    import wam.policy
    pol = wam.policy.load("wam-v1")
    env = CarEnv()
    s, _ = env.reset(CLOSED_LOOP_SEED_BLOCK)
    f, x = env.render(), s.features()
    res = {}
    for B in (1, 8, 24, 60):
        fr, st = np.repeat(f[None], B, 0), np.repeat(x[None], B, 0)
        pol.act(fr, st, "any", seed=0)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for i in range(10):
            pol.act(fr, st, "any", seed=i)
        res[B] = (time.perf_counter() - t0) / 10
        print(f"B={B:3d}: {1000 * res[B]:.1f} ms per call, {1000 * res[B] / B:.1f} ms per Chunk")
    t0 = time.perf_counter()
    for _ in range(200):
        env.render()
    print(f"render: {1000 * (time.perf_counter() - t0) / 200:.2f} ms per Frame")
    (OUT / "naive_timing.json").write_text(json.dumps(res))


def cmd_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wam.policy
    pol = wam.policy.load("wam-v1")
    eps = episodes(10)
    recs, _ = run_policy(pol, eps, k=4, keep_traj=True)
    env = CarEnv()
    tr = env.track
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (ins, d) in zip(axes.flat, COMBOS):
        for name in ("outer", "inner"):
            route = tr.route(name)
            p = np.array([route.pose_at(s)[:2] for s in np.linspace(0, route.length_m, 800)])
            ax.plot(p[:, 0], p[:, 1], color="0.8", lw=6)
        for r in recs:
            if r["instruction"] == ins and r["direction"] == d:
                p = np.array(r["traj"])
                ax.plot(p[:, 0], p[:, 1], lw=0.7)
                if "crash" in r:
                    ax.plot(r["crash"]["x"], r["crash"]["y"], "rx", ms=10, mew=2)
        ax.set_title(f"{ins}/{d}")
        ax.set_aspect("equal")
    fig.suptitle("wam-v1 closed loop, k=4, n_samples=1 (red x = Crash)")
    fig.savefig(OUT / "naive_trajectories.png", dpi=90)
    print(OUT / "naive_trajectories.png")


if __name__ == "__main__":
    {"main": cmd_main, "sweep": cmd_sweep, "timing": cmd_timing, "plot": cmd_plot}[sys.argv[1]]()
