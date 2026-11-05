"""Throwaway prototype for ticket 05 "Metrics and gates": do the four gates catch what they should?

Saboteur drivers (the clean expert with one defect) run through a closed loop on the 120 gate seeds
(`CLOSED_LOOP_SEED_BLOCK + 100_000 + k`, combination k % 6, Step 3 start rule, 900 Ticks).

    uv run python .scratch/step9-closed-loop/prototypes/gate_saboteurs.py
"""

from __future__ import annotations

import json
import sys
from itertools import product
from math import sqrt
from pathlib import Path

from toycar.dataset import CLOSED_LOOP_SEED_BLOCK, sample_render_params
from toycar.env import CarEnv
from toycar.expert import Expert, ExpertParams, score_episode, start_spec, summarize

OUT = Path(__file__).parent / "out"
COMBOS = list(product(["outer", "inner", "any"], ["cw", "ccw"]))
GATE_BASE = CLOSED_LOOP_SEED_BLOCK + 100_000
N = 120
FRAC: dict[int, float] = {}
OPP = {"outer": "inner", "inner": "outer", "any": "any"}


def gate_episodes(n=N):
    return [(GATE_BASE + k, *COMBOS[k % 6]) for k in range(n)]


def run(seed, ins, d, kind, v_cap=10.0):
    env = CarEnv()
    s, i = env.reset(seed, start_spec(ins, d), render_params=sample_render_params(seed))
    told = OPP[ins] if kind == "wrong-branch" else ins
    p = ExpertParams(burst_peak=0.0, v_cap=v_cap if kind == "slow" else 10.0,
                     p_outer=1.0 if kind == "always-outer" else 0.5)
    ex = Expert(env.track, told, p)
    ex.reset(seed, s, i)
    infos = [i]
    while True:
        a = ex.act(s, i).executed
        if kind == "zero-steer":
            a = (0.0, a[1])
        elif kind == "averaged-fork" and ins == "any":
            # outside the fork section both Branches share the line, so averaging everywhere = averaging at the Fork
            o, n = ex.control(s, "outer", d), ex.control(s, "inner", d)
            a = ((o[0] + n[0]) / 2, (o[1] + n[1]) / 2)
        r = env.step(a)
        s, i = r.state, r.info
        infos.append(i)
        if r.terminated or r.truncated:
            sc = score_episode(env.track, ins, infos)
            FRAC[id(sc)] = sc.progress_m / env.track.length_m
            return sc


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gates(sm):
    ob = sm["outer_fraction"]  # None (no scored `any` Passage) fails balance
    return {
        "crash": sm["crash_rate"] <= 0.05,
        "laps": sm["frac_laps_mean"] >= 3.5,  # fractional laps = progress_m / track.length_m
        "followed": sm["instruction_followed_rate"] >= 0.95,
        "balance": ob is not None and 0.30 <= ob <= 0.70,
    }


def main():
    eps = gate_episodes()
    kinds = [("expert", 10.0), ("zero-steer", 10.0), ("averaged-fork", 10.0),
             ("slow", 5.0), ("slow", 6.0), ("slow", 7.0), ("wrong-branch", 10.0), ("always-outer", 10.0)]
    if len(sys.argv) > 1:
        kinds = [k for k in kinds if k[0] in sys.argv[1:]]
    out = {}
    print(f"{'driver':<16} {'crash':>14} {'laps':>5} {'fLaps':>5} {'lapsMin':>7} {'followed':>16} {'wrongP':>6} {'outer':>6}  gates")
    for kind, vc in kinds:
        name = kind if kind != "slow" else f"slow(v_cap={vc:g})"
        scores = [run(sd, ins, d, kind, vc) for sd, ins, d in eps]
        sm = summarize(scores)
        sm["frac_laps_mean"] = sum(FRAC[id(sc)] for sc in scores) / len(scores)
        g = gates(sm)
        nc = sum(sc.crashed for sc in scores)
        nf = sum(sc.instruction_followed for sc in scores)
        lo, hi = wilson(nc, N)
        flo, fhi = wilson(nf, N)
        ob = sm["outer_fraction"]
        fails = [k for k, v in g.items() if not v]
        print(f"{name:<16} {sm['crash_rate']:>5.1%} [{lo:.0%}-{hi:.0%}] {sm['laps_mean']:>5.2f} {sm['frac_laps_mean']:>5.2f} {sm['laps_min']:>7} "
              f"{sm['instruction_followed_rate']:>5.1%} [{flo:.0%}-{fhi:.0%}] {sm['wrong_passages']:>6} "
              f"{(ob if ob is not None else float('nan')):>6.2f}  {'PASS' if not fails else 'FAIL ' + ','.join(fails)}")
        out[name] = {"summary": sm, "gates": g, "crash_ci": [lo, hi], "followed_ci": [flo, fhi]}
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_saboteurs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
