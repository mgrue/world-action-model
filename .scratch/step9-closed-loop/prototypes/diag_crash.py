"""Ticket 01 diagnostic: along wam-v1's own closed-loop trajectory, compare its executed actions with what the
expert's clean controller would do in the same State (Expert.control is pure). Also the model's open-loop error on
these off-distribution States vs the training-time error.

    uv run python .scratch/step9-closed-loop/prototypes/diag_crash.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from naive_loop import OUT, episodes  # noqa: E402

from toycar.dataset import sample_render_params  # noqa: E402
from toycar.env import CarEnv  # noqa: E402
from toycar.expert import Expert, ExpertParams, start_spec  # noqa: E402


def main():
    import wam.policy
    pol = wam.policy.load("wam-v1")
    eps = episodes(2)  # 12 episodes, one batch
    envs, states, infos, experts, queues, logs = [], [], [], [], [], []
    for seed, ins, d in eps:
        env = CarEnv()
        s, i = env.reset(seed, start_spec(ins, d), render_params=sample_render_params(seed))
        ex = Expert(env.track, ins, ExpertParams())
        ex.reset(seed, s, i)
        envs.append(env), states.append(s), infos.append(i), experts.append(ex), queues.append([]), logs.append([])
    alive, tick = list(range(len(eps))), 0
    while alive:
        need = [j for j in alive if not queues[j]]
        if need:
            r = pol.act(np.stack([envs[j].render() for j in need]), np.stack([states[j].features() for j in need]),
                        [eps[j][1] for j in need], seed=tick)
            for row, j in enumerate(need):
                queues[j] = [tuple(a) for a in r.actions[row, 0, :4]]
        for j in list(alive):
            s, i = states[j], infos[j]
            branch = experts[j].planned_branch
            ref = experts[j].control(s, branch, i.direction)  # expert's clean action in the model's State
            a = queues[j].pop(0)
            logs[j].append({"tick": tick, "station": i.station_m, "v": s.v, "steer": a[0], "thr": a[1],
                            "ex_steer": ref[0], "ex_thr": ref[1],
                            "lat": envs[j].track.distance_to_centerline(s.x, s.y)})
            st = envs[j].step(a)
            states[j], infos[j] = st.state, st.info
            # keep the expert's coin bookkeeping in step with the episode (redraw on section exit)
            experts[j].act(st.state, st.info)
            if st.terminated or st.truncated:
                alive.remove(j)
                logs[j][-1]["end"] = st.info.termination
        tick += 1

    allrows = [row for lg in logs for row in lg]
    st = np.array([[r["steer"], r["ex_steer"], r["thr"], r["ex_thr"], r["v"], r["lat"]] for r in allrows])
    print(f"{len(allrows)} Ticks over {len(logs)} episodes, "
          f"{sum(lg[-1].get('end') == 'crash' for lg in logs)} crashes")
    print(f"executed vs expert-clean in visited States: RMSE steer {np.sqrt(((st[:, 0] - st[:, 1]) ** 2).mean()):.3f}"
          f"  throttle {np.sqrt(((st[:, 2] - st[:, 3]) ** 2).mean()):.3f}")
    for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9)]:
        m = (st[:, 5] >= lo) & (st[:, 5] < hi)
        if m.any():
            print(f"  lat {lo:.1f}–{hi:.1f} m: n {m.sum():5d}  RMSE steer {np.sqrt(((st[m, 0] - st[m, 1]) ** 2).mean()):.3f}"
                  f"  mean steer {st[m, 0].mean():+.3f} vs expert {st[m, 1].mean():+.3f}")
    print("\nlast 40 Ticks before each Crash (every 4th): tick station v steer/expert thr/expert lat")
    for (seed, ins, d), lg in zip(eps, logs):
        if lg[-1].get("end") != "crash":
            continue
        print(f" seed {seed} {ins}/{d}")
        for r in lg[-40::4]:
            print(f"   {r['tick']:4d} {r['station']:6.1f} {r['v']:5.2f}  {r['steer']:+.2f}/{r['ex_steer']:+.2f}"
                  f"  {r['thr']:+.2f}/{r['ex_thr']:+.2f}  {r['lat']:.2f}")
    (OUT / "diag_crash.json").write_text(json.dumps(logs))


if __name__ == "__main__":
    main()
