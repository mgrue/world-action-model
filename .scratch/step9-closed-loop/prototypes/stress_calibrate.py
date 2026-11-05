"""Throwaway prototype for ticket 06 "Robustness report": calibrate push and offset-start stress against the clean expert.

Robustness seeds `CLOSED_LOOP_SEED_BLOCK + 200_000 + k`, combination k % 6, 900 Ticks.
- push(amp, ticks): from Tick 60, every 60 Ticks ±amp is added to the executed steer for `ticks` Ticks (an absolute steer
  override crashed the expert within 4 Ticks in arcs — corner geometry, not recovery)
  (sign from rng [seed, 4]); the expert keeps running underneath and resumes from the pushed State.
- offset_start(lat_lo, lat_hi, head_lo, head_hi): StartSpec with |lateral| in [lat_lo, lat_hi] m, |heading| in
  [head_lo, head_hi] deg, random signs, speed 0-6, drawn from rng [seed, 5]; redraw (≤ 50) if the pose is a Crash.
- wide_fork: TrackParams(bulge_m=7.0).

    uv run python .scratch/step9-closed-loop/prototypes/stress_calibrate.py
"""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from toycar.dataset import CLOSED_LOOP_SEED_BLOCK, PROBES, sample_render_params
from toycar.env import CarEnv
from toycar.expert import Expert, ExpertParams, score_episode, start_spec, summarize
from toycar.state import StartSpec
from toycar.track import Track

OUT = Path(__file__).parent / "out"
COMBOS = list(product(["outer", "inner", "any"], ["cw", "ccw"]))
BASE = CLOSED_LOOP_SEED_BLOCK + 200_000


def offset_spec(env, seed, ins, d, lat, head):
    rng = np.random.default_rng([seed, 5])
    base = start_spec(ins, d)
    for _ in range(50):
        sp = StartSpec(direction=d, route=base.route if base.route else ["outer", "inner"][int(rng.integers(2))],
                       station_m=float(rng.uniform(0, env.track.length_m)),
                       lateral_m=float(rng.choice([-1, 1]) * rng.uniform(*lat)),
                       heading_offset=float(rng.choice([-1, 1]) * math.radians(rng.uniform(*head))),
                       speed=float(rng.uniform(0, 6)))
        try:
            s, _ = env.reset(seed, sp)
            return sp, 0
        except ValueError:
            continue
    raise RuntimeError("no valid offset start")


def run(seed, ins, d, cond):
    kind = cond[0]
    track = Track(PROBES["wide_fork"].track_params) if kind == "wide_fork" else None
    env = CarEnv(track=track)
    sp = start_spec(ins, d)
    if kind == "offset":
        sp, _ = offset_spec(env, seed, ins, d, cond[1], cond[2])
    s, i = env.reset(seed, sp, render_params=sample_render_params(seed))
    ex = Expert(env.track, ins, ExpertParams(burst_peak=0.0))
    ex.reset(seed, s, i)
    rng = np.random.default_rng([seed, 4])
    infos, lat, push_crash, peaks = [i], [], 0, []
    sign, push_end, push_start = 1.0, -1, -1
    t = 0
    while True:
        a = ex.act(s, i).executed
        if kind == "push":
            amp, n = cond[1], cond[2]
            if t >= 60 and (t - 60) % 60 == 0:
                sign, push_start, push_end = float(rng.choice([-1, 1])), t, t + n
            if t < push_end:
                a = (min(max(a[0] + sign * amp, -1.0), 1.0), a[1])  # additive offset on the driver's steer
        r = env.step(a)
        s, i = r.state, r.info
        t += 1
        infos.append(i)
        lat.append(env.track.distance_to_centerline(s.x, s.y))
        if r.terminated or r.truncated:
            sc = score_episode(env.track, ins, infos)
            if sc.crashed and push_start >= 0 and t - push_start <= 45:
                push_crash = 1
            return sc, float(np.mean(lat)), float(np.max(lat)), push_crash


def main(n=60):
    eps = [(BASE + k, *COMBOS[k % 6]) for k in range(n)]
    conds = [("none",), ("wide_fork",),
             ("push", 0.3, 8), ("push", 0.4, 8), ("push", 0.5, 8), ("push", 0.6, 10), ("push", 0.8, 10),
             ("offset", (1.0, 1.5), (10, 20)), ("offset", (1.0, 1.5), (10, 15)), ("offset", (1.0, 1.25), (10, 15))]
    out = {}
    print(f"{'condition':<28} {'crash':>6} {'fLaps':>6} {'follow':>7} {'outer':>6} {'latMean':>8} {'latMax':>7} pushCrash")
    for c in conds:
        rows = [run(sd, ins, d, c) for sd, ins, d in eps]
        sm = summarize([r[0] for r in rows])
        name = " ".join(str(x) for x in c)
        fl = np.mean([r[0].progress_m for r in rows]) / 88.056
        ob = sm["outer_fraction"]
        print(f"{name:<28} {sm['crash_rate']:>6.1%} {fl:>6.2f} {sm['instruction_followed_rate']:>7.1%} "
              f"{(ob if ob is not None else float('nan')):>6.2f} {np.mean([r[1] for r in rows]):>8.2f} "
              f"{np.max([r[2] for r in rows]):>7.2f} {sum(r[3] for r in rows)}")
        out[name] = {"summary": sm, "frac_laps": fl}
    OUT.mkdir(exist_ok=True)
    (OUT / "stress_calibrate.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
